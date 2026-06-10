"""
@file crn.py
@brief Counterfactual Recurrent Network (CRN) for treatment-effect estimation on panel data.

Trains a GRU/LSTM encoder with a gradient-reversal layer to produce
treatment-invariant representations, then predicts factual and counterfactual
outcomes at every visit step.

Run from the repo root:
    python -m src.crn.crn

Inputs  : src/crn/preprocessed/train.csv, test.csv, meta.json, scaler.pkl, target_scaler.pkl
Outputs : src/crn/preprocessed/train_mixed.csv, test_mixed.csv, crn_weights.pt
"""

import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


INPUT_DIR    = "src/crn/preprocessed"
OUTPUT_DIR   = "src/crn/preprocessed"
RANDOM_SEED  = 42
N_TREATMENTS = 3

HIDDEN_DIM   = 64
N_LAYERS     = 1
DROPOUT      = 0.1
ALPHA        = 1.0
LAMBDA_BAL   = 0.5

EPOCHS       = 150
BATCH_SIZE   = 32
LR           = 1e-3
PATIENCE     = 12

TARGET_COL    = "time_to_last_days"
TREATMENT_COL = "Chemotherapieregime_enc"

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _GRLFn(torch.autograd.Function):
    """
    @brief Gradient Reversal Function: identity in forward, negated gradient in backward.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class GradReversal(nn.Module):
    """
    @brief Gradient Reversal Layer used for adversarial treatment-balancing.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @brief Apply gradient reversal to x.

        @param x  Input tensor.
        @return   x unchanged in forward; gradient is negated in backward.
        """
        return _GRLFn.apply(x, self.alpha)

    def set_alpha(self, alpha: float):
        """
        @brief Update the reversal strength.

        @param alpha  New alpha value.
        """
        self.alpha = alpha


class CRN(nn.Module):
    """
    @brief Counterfactual Recurrent Network (CRN).

    Encoder input per time step: [x_t | one_hot(a_{t-1}) | y_{t-1}]
      x_t     = normalised clinical features
      a_{t-1} = treatment at previous step (0 at t=0)
      y_{t-1} = normalised outcome at previous step (0 at t=0)

    Outcome prediction: [H_t | one_hot(a_t)] -> MLP -> y_hat_t
    By conditioning the outcome head on a_t, H_t remains treatment-invariant;
    for counterfactuals a_t is simply swapped.

    Balancing: GRL(H_t) -> treatment_head -> CE-loss against a_t
    The GRL reverses gradients so the encoder learns NOT to encode a_t.
    """

    def __init__(
        self,
        input_dim: int,
        n_treatments: int,
        hidden_dim: int = 64,
        n_layers: int = 1,
        dropout: float = 0.1,
        alpha: float = 1.0,
    ):
        """
        @brief Initialise the CRN.

        @param input_dim     Number of clinical feature dimensions.
        @param n_treatments  Number of treatment arms.
        @param hidden_dim    LSTM hidden state size.
        @param n_layers      Number of LSTM layers.
        @param dropout       Dropout probability (applied between LSTM layers and in heads).
        @param alpha         Gradient reversal strength.
        """
        super().__init__()
        self.n_treatments = n_treatments
        self.hidden_dim   = hidden_dim
        self.n_layers     = n_layers

        enc_dim = input_dim + n_treatments + 1  # x + one_hot(a_prev) + y_prev
        self.encoder = nn.LSTM(
            enc_dim, hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.grl = GradReversal(alpha)
        self.treatment_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_treatments),
        )

        self.outcome_head = nn.Sequential(
            nn.Linear(hidden_dim + n_treatments, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _encode(
        self,
        x: torch.Tensor,
        prev_treatments: torch.Tensor,
        prev_outcomes: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        @brief Run the LSTM encoder over the input sequence.

        @param x               Feature tensor (B, T, input_dim).
        @param prev_treatments Previous-step treatment indices (B, T).
        @param prev_outcomes   Previous-step outcomes (B, T).
        @param mask            Boolean mask of real steps (B, T) or None.
        @return Tuple (H, h_n, c_n) - hidden states and final cell state.
        """
        B, T, _ = x.shape
        a_oh   = F.one_hot(prev_treatments.clamp(0), self.n_treatments).float()
        y_in   = prev_outcomes.unsqueeze(-1)
        enc_in = torch.cat([x, a_oh, y_in], dim=-1)

        if mask is not None:
            lengths = mask.sum(1).cpu().clamp(min=1)
            packed  = nn.utils.rnn.pack_padded_sequence(
                enc_in, lengths, batch_first=True, enforce_sorted=False
            )
            out_p, (h_n, c_n) = self.encoder(packed)
            H, _ = nn.utils.rnn.pad_packed_sequence(
                out_p, batch_first=True, total_length=T
            )
        else:
            H, (h_n, c_n) = self.encoder(enc_in)

        return H, h_n, c_n

    def forward(
        self,
        x: torch.Tensor,
        treatments: torch.Tensor,
        prev_treatments: torch.Tensor,
        prev_outcomes: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        @brief Training forward pass.

        @param x               Feature tensor (B, T, input_dim).
        @param treatments      Current-step treatment indices (B, T).
        @param prev_treatments Previous-step treatment indices (B, T).
        @param prev_outcomes   Previous-step outcomes (B, T).
        @param mask            Boolean mask of real steps (B, T) or None.
        @return Tuple (y_pred (B, T), t_logits (B, T, n_treatments)).
        """
        H, _, _ = self._encode(x, prev_treatments, prev_outcomes, mask)

        a_oh   = F.one_hot(treatments.clamp(0), self.n_treatments).float()
        y_pred = self.outcome_head(torch.cat([H, a_oh], dim=-1)).squeeze(-1)

        t_logits = self.treatment_head(self.grl(H))

        return y_pred, t_logits

    @torch.no_grad()
    def predict_counterfactual(
        self,
        x: torch.Tensor,
        prev_treatments: torch.Tensor,
        prev_outcomes: torch.Tensor,
        mask: torch.Tensor,
        target_treatment: int,
    ) -> torch.Tensor:
        """
        @brief Predict outcomes under a hypothetical treatment for all time steps.

        @param x                Feature tensor (B, T, input_dim).
        @param prev_treatments  Previous-step treatment indices (B, T).
        @param prev_outcomes    Previous-step outcomes (B, T).
        @param mask             Boolean mask of real steps (B, T).
        @param target_treatment Counterfactual treatment arm to apply at every step.
        @return Counterfactual outcomes y_cf (B, T).
        """
        H, _, _ = self._encode(x, prev_treatments, prev_outcomes, mask)
        B, T, _ = H.shape
        t_cf = torch.full((B, T), target_treatment, dtype=torch.long, device=H.device)
        a_oh = F.one_hot(t_cf, self.n_treatments).float()
        return self.outcome_head(torch.cat([H, a_oh], dim=-1)).squeeze(-1)


class CRNDataset(Dataset):
    """
    @brief PyTorch Dataset that builds padded patient sequences from a preprocessed DataFrame.

    Each sample contains:
      x               (max_seq, n_features)  - normalised features
      treatments      (max_seq,)             - treatment at t
      prev_treatments (max_seq,)             - treatment at t-1 (0 at t=0)
      prev_outcomes   (max_seq,)             - normalised outcome at t-1 (0 at t=0)
      outcomes        (max_seq,)             - normalised outcome at t
      mask            (max_seq,)             - True = real time step
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        subscale_cols: list,
        scaler: StandardScaler,
        target_scaler: StandardScaler,
        max_seq_len: int,
    ):
        """
        @brief Build all patient sequence samples from the DataFrame.

        @param df            Preprocessed panel DataFrame sorted by visit.
        @param feature_cols  Ordered list of model input feature column names.
        @param subscale_cols Subscale columns that receive StandardScaler normalisation.
        @param scaler        Fitted feature scaler (subscales only).
        @param target_scaler Fitted target scaler.
        @param max_seq_len   Maximum sequence length (longer sequences are truncated).
        """
        self.max_seq      = max_seq_len
        self.feature_cols = feature_cols
        self.samples: list[dict] = []

        other_cols = [c for c in feature_cols if c not in subscale_cols]

        for pid, grp in df.groupby("Patienten_ID"):
            grp = grp.sort_values("visit_nr")
            T   = min(len(grp), max_seq_len)

            x_subscales = scaler.transform(
                grp[subscale_cols].values[:T].astype(np.float32)
            )
            x_other = grp[other_cols].values[:T].astype(np.float32) if other_cols else np.empty((T, 0), dtype=np.float32)
            x_sc = np.hstack([x_subscales, x_other])
            y_sc = target_scaler.transform(
                grp[TARGET_COL].values[:T].reshape(-1, 1).astype(np.float32)
            ).flatten()

            t_arr  = grp[TREATMENT_COL].values[:T].astype(np.int64)
            pt_arr = np.concatenate([[0], t_arr[:-1]])
            py_arr = np.concatenate([[0.0], y_sc[:-1]])

            F_dim  = x_sc.shape[1]
            x_pad  = np.zeros((max_seq_len, F_dim), dtype=np.float32)
            y_pad  = np.zeros(max_seq_len, dtype=np.float32)
            t_pad  = np.zeros(max_seq_len, dtype=np.int64)
            pt_pad = np.zeros(max_seq_len, dtype=np.int64)
            py_pad = np.zeros(max_seq_len, dtype=np.float32)
            mask   = np.zeros(max_seq_len, dtype=bool)

            x_pad[:T]  = x_sc
            y_pad[:T]  = y_sc
            t_pad[:T]  = t_arr
            pt_pad[:T] = pt_arr
            py_pad[:T] = py_arr
            mask[:T]   = True

            self.samples.append({
                "pid":             pid,
                "x":               x_pad,
                "treatments":      t_pad,
                "prev_treatments": pt_pad,
                "prev_outcomes":   py_pad,
                "outcomes":        y_pad,
                "mask":            mask,
                "y_raw":           grp[TARGET_COL].values[:T].astype(np.float32),
                "t_raw":           t_arr,
                "seq_len":         T,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        @brief Return one patient sample as a dict of tensors.

        @param idx  Sample index.
        @return Dict with keys x, treatments, prev_treatments, prev_outcomes, outcomes, mask.
        """
        s = self.samples[idx]
        return {k: torch.tensor(s[k]) for k in
                ["x", "treatments", "prev_treatments", "prev_outcomes", "outcomes", "mask"]}


def crn_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    t_logits: torch.Tensor,
    t_true: torch.Tensor,
    mask: torch.Tensor,
    lambda_bal: float,
) -> tuple[torch.Tensor, float, float]:
    """
    @brief Compute CRN combined loss: MSE outcome + lambda_bal * CE treatment.

    Both terms are computed only over real time steps (mask == True).

    @param y_pred      Predicted outcomes (B, T).
    @param y_true      Ground-truth outcomes (B, T).
    @param t_logits    Treatment logits from the balancing head (B, T, n_treatments).
    @param t_true      Ground-truth treatment indices (B, T).
    @param mask        Boolean mask of real steps (B, T).
    @param lambda_bal  Weight on the balancing cross-entropy term.
    @return Tuple (total_loss, outcome_mse_scalar, balancing_ce_scalar).
    """
    m = mask.reshape(-1)

    y_pred_m = y_pred.reshape(-1)[m]
    y_true_m = y_true.reshape(-1)[m]
    loss_out = F.mse_loss(y_pred_m, y_true_m)

    t_logits_m = t_logits.reshape(-1, N_TREATMENTS)[m]
    t_true_m   = t_true.reshape(-1)[m]
    loss_bal   = F.cross_entropy(t_logits_m, t_true_m)

    total = loss_out + lambda_bal * loss_bal
    return total, loss_out.item(), loss_bal.item()


def run_epoch(
    model: CRN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    lambda_bal: float,
    device: torch.device,
    train: bool = True,
) -> tuple[float, float, float]:
    """
    @brief Run one full epoch of training or evaluation.

    @param model       CRN model.
    @param loader      DataLoader for this epoch.
    @param optimizer   Adam optimizer (None for eval mode).
    @param lambda_bal  Balancing loss weight.
    @param device      Torch device.
    @param train       If True, compute gradients and update weights.
    @return Tuple (mean_total_loss, mean_outcome_mse, mean_balancing_ce).
    """
    model.train(train)
    ctx    = torch.enable_grad() if train else torch.no_grad()
    totals = [0.0, 0.0, 0.0]

    with ctx:
        for batch in loader:
            x   = batch["x"].to(device)
            t   = batch["treatments"].to(device)
            pt  = batch["prev_treatments"].to(device)
            py  = batch["prev_outcomes"].to(device)
            y   = batch["outcomes"].to(device)
            msk = batch["mask"].to(device)

            if train:
                optimizer.zero_grad()

            y_pred, t_logits = model(x, t, pt, py, msk)
            loss, lo, lb     = crn_loss(y_pred, y, t_logits, t, msk, lambda_bal)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            totals[0] += loss.item()
            totals[1] += lo
            totals[2] += lb

    n = max(len(loader), 1)
    return totals[0] / n, totals[1] / n, totals[2] / n


def train_crn(
    model: CRN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    lambda_bal: float,
    patience: int,
    device: torch.device,
) -> CRN:
    """
    @brief Train the CRN with early stopping on validation loss.

    @param model        CRN model to train.
    @param train_loader DataLoader for training data.
    @param val_loader   DataLoader for validation data.
    @param epochs       Maximum number of training epochs.
    @param lr           Initial learning rate for Adam.
    @param lambda_bal   Balancing loss weight.
    @param patience     Early-stopping patience in epochs.
    @param device       Torch device.
    @return CRN with best validation-loss weights loaded.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience // 2, factor=0.5
    )

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    print(f"  Training on {device} for max {epochs} epochs (patience={patience})")
    print(f"  {'Epoch':>6}  {'Train-Loss':>10}  {'Val-Loss':>9}  "
          f"{'Out-MSE':>8}  {'Bal-CE':>7}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*9}  {'─'*8}  {'─'*7}")

    for ep in range(1, epochs + 1):
        tl, to, tb = run_epoch(model, train_loader, optimizer, lambda_bal, device, train=True)
        vl, vo, vb = run_epoch(model, val_loader,   None,      lambda_bal, device, train=False)
        scheduler.step(vl)

        if ep % 10 == 0 or ep == 1:
            print(f"  {ep:>6}  {tl:>10.4f}  {vl:>9.4f}  {vo:>8.4f}  {vb:>7.4f}")

        if vl < best_val_loss - 1e-5:
            best_val_loss = vl
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {ep} (best val loss: {best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)
    return model


def generate_counterfactuals(
    model: CRN,
    dataset: CRNDataset,
    df_orig: pd.DataFrame,
    target_scaler: StandardScaler,
    device: torch.device,
    n_treatments: int,
) -> pd.DataFrame:
    """
    @brief Generate Y0, Y1, Y2 counterfactual outcomes for every patient x time step.

    Rule: Y_t_factual = true observed value; Y_t_cf = CRN prediction under arm t'.

    @param model          Trained CRN model.
    @param dataset        CRNDataset built from df_orig.
    @param df_orig        Original panel DataFrame (used as base for the output).
    @param target_scaler  Fitted target scaler for inverse-transforming predictions.
    @param device         Torch device.
    @param n_treatments   Number of treatment arms.
    @return df_orig copy with additional columns Y0, Y1, ... Y{n_treatments-1}.
    """
    model.eval()

    pid_results: dict[int | str, dict] = {}

    for sample in dataset.samples:
        pid = sample["pid"]
        T   = sample["seq_len"]

        x      = torch.tensor(sample["x"]).unsqueeze(0).to(device)
        prev_t = torch.tensor(sample["prev_treatments"]).unsqueeze(0).to(device)
        prev_y = torch.tensor(sample["prev_outcomes"]).unsqueeze(0).to(device)
        msk    = torch.tensor(sample["mask"]).unsqueeze(0).to(device)

        y_cfs = {}
        for t_cf in range(n_treatments):
            y_sc  = model.predict_counterfactual(x, prev_t, prev_y, msk, t_cf)
            y_sc  = y_sc[0, :T].cpu().numpy().reshape(-1, 1)
            y_day = target_scaler.inverse_transform(y_sc).flatten()
            y_day = np.clip(y_day, 0, None).round().astype(int)
            y_cfs[f"Y{t_cf}"] = y_day

        pid_results[pid] = y_cfs

    df_out = df_orig.copy()
    for t_cf in range(n_treatments):
        df_out[f"Y{t_cf}"] = np.nan

    for pid, grp in df_out.groupby("Patienten_ID"):
        if pid not in pid_results:
            continue
        idx = grp.sort_values("visit_nr").index
        for col, vals in pid_results[pid].items():
            T_actual = min(len(idx), len(vals))
            df_out.loc[idx[:T_actual], col] = vals[:T_actual]

    for t_cf in range(n_treatments):
        df_out[f"Y{t_cf}"] = df_out[f"Y{t_cf}"].fillna(0).astype(int)

    return df_out


def validate_counterfactuals(df_mixed: pd.DataFrame, n_treatments: int = N_TREATMENTS):
    """
    @brief Run sanity checks on the generated counterfactual DataFrame.

    Prints factual-consistency mismatches, mean outcomes, and mean CATE.

    @param df_mixed      DataFrame with Y0...Y{k} columns appended.
    @param n_treatments  Number of treatment arms.
    """
    print("\n  Factual consistency (Y_t_actual == real Y):")
    for t in range(n_treatments):
        mask     = df_mixed[TREATMENT_COL] == t
        real_y   = df_mixed.loc[mask, TARGET_COL]
        cf_y     = df_mixed.loc[mask, f"Y{t}"]
        mismatch = (real_y.values.astype(int) != cf_y.values).sum()
        print(f"    Arm {t}: {mask.sum()} rows  mismatch={mismatch}")

    print("\n  Mean outcomes (days):")
    for t in range(n_treatments):
        col = df_mixed[f"Y{t}"]
        print(f"    Y{t}: {col.mean():.1f} ± {col.std():.1f}  [{col.min()}, {col.max()}]")

    print("\n  Mean CATE (Y_t − Y0):")
    for t in range(1, n_treatments):
        cate = (df_mixed[f"Y{t}"] - df_mixed["Y0"]).mean()
        print(f"    E[Y{t}−Y0] = {cate:+.1f} days")


def load_artifacts(input_dir: str) -> tuple:
    """
    @brief Load preprocessed CSV files, metadata, and scalers from disk.

    @param input_dir  Directory containing train.csv, test.csv, meta.json,
                      scaler.pkl, target_scaler.pkl.
    @return Tuple (df_train, df_test, meta_dict, scaler, target_scaler).
    """
    df_train = pd.read_csv(f"{input_dir}/train.csv")
    df_test  = pd.read_csv(f"{input_dir}/test.csv")
    meta     = json.load(open(f"{input_dir}/meta.json"))

    with open(f"{input_dir}/scaler.pkl",        "rb") as f: scaler        = pickle.load(f)
    with open(f"{input_dir}/target_scaler.pkl",  "rb") as f: target_scaler = pickle.load(f)

    return df_train, df_test, meta, scaler, target_scaler


def build_dataloaders(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list,
    subscale_cols: list,
    scaler: StandardScaler,
    target_scaler: StandardScaler,
    max_seq_len: int,
    val_frac: float = 0.15,
) -> tuple[DataLoader, DataLoader, DataLoader, CRNDataset, CRNDataset]:
    """
    @brief Build train, validation, and test DataLoaders from panel DataFrames.

    @param df_train      Full training panel DataFrame.
    @param df_test       Test panel DataFrame.
    @param feature_cols  Ordered feature column names.
    @param subscale_cols Subscale columns to normalise.
    @param scaler        Fitted feature scaler.
    @param target_scaler Fitted target scaler.
    @param max_seq_len   Maximum sequence length for padding.
    @param val_frac      Fraction of training patients held out for validation.
    @return Tuple (train_loader, val_loader, test_loader, train_dataset, test_dataset).
    """
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(1, test_size=val_frac, random_state=RANDOM_SEED)
    tr_idx, val_idx = next(gss.split(df_train, groups=df_train["Patienten_ID"]))
    df_tr  = df_train.iloc[tr_idx]
    df_val = df_train.iloc[val_idx]

    ds_tr  = CRNDataset(df_tr,   feature_cols, subscale_cols, scaler, target_scaler, max_seq_len)
    ds_val = CRNDataset(df_val,  feature_cols, subscale_cols, scaler, target_scaler, max_seq_len)
    ds_te  = CRNDataset(df_test, feature_cols, subscale_cols, scaler, target_scaler, max_seq_len)

    ldr_tr  = DataLoader(ds_tr,  batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    ldr_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
    ldr_te  = DataLoader(ds_te,  batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Train split : {len(ds_tr)} patients")
    print(f"  Val split   : {len(ds_val)} patients")
    print(f"  Test        : {len(ds_te)} patients")

    return ldr_tr, ldr_val, ldr_te, ds_tr, ds_te


def save_results(
    df_train_mixed: pd.DataFrame,
    df_test_mixed: pd.DataFrame,
    model: CRN,
    output_dir: str,
) -> None:
    """
    @brief Save train_mixed.csv, test_mixed.csv, and model weights to disk.

    @param df_train_mixed  Training DataFrame with counterfactual columns.
    @param df_test_mixed   Test DataFrame with counterfactual columns.
    @param model           Trained CRN model.
    @param output_dir      Directory path to write outputs.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_train_mixed.to_csv(f"{output_dir}/train_mixed.csv", index=False)
    df_test_mixed.to_csv(f"{output_dir}/test_mixed.csv",   index=False)
    torch.save(model.state_dict(), f"{output_dir}/crn_weights.pt")
    print(f"  train_mixed.csv  ({len(df_train_mixed)} rows)")
    print(f"  test_mixed.csv   ({len(df_test_mixed)} rows)")
    print(f"  crn_weights.pt")


def main():
    """
    @brief Full CRN pipeline: load -> build dataloaders -> train -> generate counterfactuals -> save.
    """
    df_train, df_test, meta, scaler, target_scaler = load_artifacts(INPUT_DIR)
    feature_cols  = meta["feature_cols"]
    max_seq       = meta["max_seq_len"]
    input_dim     = len(feature_cols)
    subscale_cols = meta["subscale_cols"]
    print(f"  Features: {input_dim}  |  max_seq: {max_seq}  |  Treatments: {N_TREATMENTS}")

    ldr_tr, ldr_val, ldr_te, ds_tr, ds_te = build_dataloaders(
        df_train, df_test, feature_cols, subscale_cols, scaler, target_scaler, max_seq
    )

    print(f"\nInitialising CRN  (hidden={HIDDEN_DIM}, layers={N_LAYERS}, alpha={ALPHA})")
    model = CRN(
        input_dim    = input_dim,
        n_treatments = N_TREATMENTS,
        hidden_dim   = HIDDEN_DIM,
        n_layers     = N_LAYERS,
        dropout      = DROPOUT,
        alpha        = ALPHA,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    print(f"\nTraining  (lambda_bal={LAMBDA_BAL}, lr={LR}, batch={BATCH_SIZE})")
    model = train_crn(
        model, ldr_tr, ldr_val,
        epochs=EPOCHS, lr=LR, lambda_bal=LAMBDA_BAL,
        patience=PATIENCE, device=DEVICE,
    )

    tl, to, tb = run_epoch(model, ldr_te, None, LAMBDA_BAL, DEVICE, train=False)
    print(f"  Test loss: {tl:.4f}  |  Outcome MSE: {to:.4f}  |  Bal CE: {tb:.4f}")

    df_train_mixed = generate_counterfactuals(
        model, ds_tr, df_train, target_scaler, DEVICE, N_TREATMENTS
    )
    df_test_mixed = generate_counterfactuals(
        model, ds_te, df_test, target_scaler, DEVICE, N_TREATMENTS
    )

    validate_counterfactuals(df_test_mixed)
    save_results(df_train_mixed, df_test_mixed, model, OUTPUT_DIR)


if __name__ == "__main__":
    main()
