"""
@file semisynth_gen.py
@brief Generate semi-synthetic potential outcomes Y_0...Y_{K-1} per visit,
       calibrated to the real time_to_last_days distribution.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.preprocess.preprocess import OUTPUT_DIR, SUBSCALE_COLS, TARGET_COL, TREATMENT_COL, TOP_N_REGIMES


RANDOM_STATE = 42


GAMMA_PREV_WEIGHT = 0.55
GAMMA_BASELINE_WEIGHT = 0.30
GAMMA_NOISE = 0.20

GAMMA_SHAPE = 0.7


CARRY: dict = {0: 0.00, 1: 0.20, 2: -0.15}
BASE_OFFSETS: dict = {0: 0.00, 1: 0.10, 2: 0.05}
TREATMENT_STRENGTH = 0.35


TREATMENT_SUBSCALE_WEIGHTS: dict = {
    0: {
        "SUBSCALE_Pain":                 +0.6,
        "SUBSCALE_Physical_Functioning": +0.6,
    },
    1: {
        "SUBSCALE_Pain":                 -0.6,
        "SUBSCALE_Physical_Functioning": -0.5,
        "SUBSCALE_Emotional_Functioning": +0.4,
    },
    2: {
        "SUBSCALE_Fatigue":            -0.7,
        "SUBSCALE_Nausea___Vomiting":  -0.5,
        "SUBSCALE_Sleep_Disturbances": -0.4,
    },
}


HEALTH_SCORE_WEIGHTS: dict = {
    "SUBSCALE_Physical_Functioning":   0.40,
    "SUBSCALE_Role_Functioning":       0.25,
    "SUBSCALE_Emotional_Functioning":  0.20,
    "SUBSCALE_Cognitive_Functioning":  0.10,
    "SUBSCALE_Global_Quality_of_Life": 0.35,
    "SUBSCALE_Fatigue":               -0.30,
    "SUBSCALE_Nausea___Vomiting":     -0.25,
    "SUBSCALE_Pain":                  -0.35,
    "SUBSCALE_Sleep_Disturbances":    -0.10,
    "SUBSCALE_Appetite_Loss":         -0.10,
}


def _full_name(short: str) -> str:
    """
    @brief Convert a short subscale name to the full EORTC column name.

    @param short  Short subscale identifier, e.g. 'SUBSCALE_Pain'.
    @return Full EORTC column name.
    """
    return f"EORTC_QLQ_C30__3_1__{short}"


def _build_health_weight_vector(feature_cols: list) -> np.ndarray:
    """
    @brief Build a weight vector aligned to feature_cols for baseline health scoring.

    @param feature_cols  Full EORTC column names used as model features.
    @return 1-D weight array of length len(feature_cols).
    """
    full_weights = {_full_name(k): v for k, v in HEALTH_SCORE_WEIGHTS.items()}
    return np.array([full_weights.get(c, 0.0) for c in feature_cols])


def _subscale_offset(treatment_k: int, x_scaled_row: np.ndarray,
                     feature_cols: list) -> float:
    """
    @brief Compute the subscale-dependent log-mean shift for treatment_k at a visit.

    The offset is added to the log-mean of the gamma outcome distribution.
    Positive offset means a larger expected outcome.

    @param treatment_k    Treatment index.
    @param x_scaled_row   Standardised feature values for one row.
    @param feature_cols   Full EORTC column names matching x_scaled_row.
    @return Scalar log-scale offset.
    """
    weights    = TREATMENT_SUBSCALE_WEIGHTS.get(treatment_k, {})
    col_to_idx = {c: i for i, c in enumerate(feature_cols)}
    offset     = BASE_OFFSETS.get(treatment_k, 0.0)
    for short, sign in weights.items():
        full = _full_name(short)
        if full in col_to_idx:
            offset += TREATMENT_STRENGTH * sign * float(x_scaled_row[col_to_idx[full]])
    return offset


def _real_outcome_stats(real_y: np.ndarray) -> tuple:
    """
    @brief Compute the log-mean and log-std of real outcomes for calibration.

    A small positive shift is added before the log to handle real_y == 0 rows
    (early dropouts), so that those rows remain representable on a log scale.

    @param real_y  Array of real time_to_last_days outcomes.
    @return Tuple (log_mean, log_std) used as the target for synthetic outcomes.
    """
    shifted = real_y.astype(float) + 1.0
    log_y   = np.log(shifted)
    return float(log_y.mean()), float(log_y.std())


def generate_synthetic_outcomes(df: pd.DataFrame, n_treatments: int,
                                seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    @brief Generate calibrated semi-synthetic potential outcomes per visit.

    Per-visit outcome is drawn from a gamma distribution whose mean depends on
    a hidden log-scale state h. The state evolves over visits using three
    inputs: the baseline health score from the patient's subscales, the
    previous factual outcome in the sequence, and a treatment-specific carry
    term. At visit 0 the state is initialised from baseline subscales only.

    Outcomes are calibrated post-hoc by mapping the synthetic factual
    log-distribution to the real log-distribution of time_to_last_days, so
    that the marginal scale and shape of the synthetic outcomes match the
    real data including the early-dropout left tail.

    @param df            Preprocessed split with SUBSCALE_COLS, TREATMENT_COL,
                         TARGET_COL, 'Patienten_ID', 'examdate'.
    @param n_treatments  Number of treatments (= TOP_N_REGIMES).
    @param seed          Random seed.
    @return df with n_treatments additional columns Y0...Y{n_treatments-1}.
    """
    rng          = np.random.default_rng(seed)
    feature_cols = [c for c in SUBSCALE_COLS if c in df.columns]
    w_vec        = _build_health_weight_vector(feature_cols)

    df = df.copy().reset_index(drop=True)
    df["examdate"] = pd.to_datetime(df["examdate"], errors="coerce")
    df = df.sort_values(["Patienten_ID", "examdate"]).reset_index(drop=True)

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].values)

    real_y     = df[TARGET_COL].astype(float).values
    real_log_y = np.log(real_y + 1.0)
    real_log_z = (real_log_y - real_log_y.mean()) / max(1e-6, real_log_y.std())

    t_arr          = df[TREATMENT_COL].astype(int).values
    patient_ids    = df["Patienten_ID"].values
    all_treatments = list(range(n_treatments))

    N       = len(df)
    Y_model = {k: np.zeros(N) for k in all_treatments}

    for pid in pd.unique(patient_ids):
        mask  = np.where(patient_ids == pid)[0]
        x_pat = X_scaled[mask]
        t_pat = t_arr[mask]
        y_pat = real_y[mask]

        baseline_score   = float(x_pat[0] @ w_vec)
        patient_anchor   = float(real_log_z[mask].mean())  # NEW

        h = (
            GAMMA_BASELINE_WEIGHT * baseline_score
        + GAMMA_PREV_WEIGHT     * patient_anchor          # NEW
        + rng.normal(0.0, GAMMA_NOISE)
        )

        for v, global_idx in enumerate(mask):
            for k in all_treatments:
                offset = _subscale_offset(k, x_pat[v], feature_cols)
                mu_log = h + offset
                Y_model[k][global_idx] = float(
                    rng.gamma(shape=GAMMA_SHAPE,
                              scale=np.exp(mu_log) / GAMMA_SHAPE)
                )

            if v + 1 < len(mask):
                next_baseline = float(x_pat[v + 1] @ w_vec)
                prev_y_z      = float(real_log_z[global_idx])
                h = (
                    GAMMA_BASELINE_WEIGHT * next_baseline
                    + GAMMA_PREV_WEIGHT   * prev_y_z
                    + CARRY.get(int(t_pat[v]), 0.0)
                    + rng.normal(0.0, GAMMA_NOISE)
                )

    Y_obs = np.array([Y_model[t_arr[i]][i] for i in range(N)])

    src_q = np.percentile(Y_obs, [10, 50, 90])
    tgt_q = np.percentile(real_y, [10, 50, 90])

    for k in all_treatments:
        Y_model[k] = np.interp(
            Y_model[k],
            np.concatenate([[0], src_q, [Y_obs.max() + 1]]),
            np.concatenate([[0], tgt_q, [real_y.max() + 1]]),
        )
    Y_model[k] = np.clip(Y_model[k], 0.0, 1500.0)

    for k in all_treatments:
        df[f"Y{k}"] = Y_model[k]

    return df


def build_mixed_csv(df: pd.DataFrame, n_treatments: int, seed: int) -> pd.DataFrame:
    """
    @brief Attach synthetic Y columns to a preprocessed split and enforce column order.

    @param df            Preprocessed split (train or test).
    @param n_treatments  Number of treatments.
    @param seed          Random seed passed to generate_synthetic_outcomes.
    @return DataFrame ordered by [Patienten_ID, examdate] with required columns.
    """
    df = generate_synthetic_outcomes(df, n_treatments=n_treatments, seed=seed)

    static_cols      = ["Patienten_ID", "examdate", TARGET_COL, TREATMENT_COL]
    subscale_present = [c for c in SUBSCALE_COLS if c in df.columns]
    extra_cols       = [c for c in ["Geschlecht_enc", "Initiale_Diagnose_enc", "visit_nr"]
                        if c in df.columns]
    y_cols           = [f"Y{k}" for k in range(n_treatments)]

    ordered = static_cols + subscale_present + extra_cols + y_cols
    ordered = [c for c in ordered if c in df.columns]
    return df[ordered]


def main():
    """
    @brief Load preprocessed splits, generate synthetic outcomes, write mixed CSVs.

    Reads <OUTPUT_DIR>/train.csv and <OUTPUT_DIR>/test.csv. Writes
    train_mixed.csv and test_mixed.csv to the same directory.
    """
    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    test_path  = os.path.join(OUTPUT_DIR, "test.csv")

    print(f"[mixed] Loading {train_path}")
    train_df = pd.read_csv(train_path, low_memory=False)
    print(f"  {len(train_df)} rows  |  {train_df['Patienten_ID'].nunique()} patients")

    print(f"[mixed] Loading {test_path}")
    test_df = pd.read_csv(test_path, low_memory=False)
    print(f"  {len(test_df)} rows  |  {test_df['Patienten_ID'].nunique()} patients")

    n_treatments = TOP_N_REGIMES

    print(f"\n[mixed] Generating {n_treatments} synthetic outcomes for train split …")
    train_mixed = build_mixed_csv(train_df, n_treatments=n_treatments, seed=RANDOM_STATE)

    print(f"[mixed] Generating {n_treatments} synthetic outcomes for test split …")
    test_mixed  = build_mixed_csv(test_df,  n_treatments=n_treatments, seed=RANDOM_STATE + 1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_out = os.path.join(OUTPUT_DIR, "train_mixed.csv")
    test_out  = os.path.join(OUTPUT_DIR, "test_mixed.csv")

    train_mixed.to_csv(train_out, index=False)
    test_mixed.to_csv(test_out,   index=False)

    print(f"\n[mixed] Written: {train_out}  ({len(train_mixed)} rows, {len(train_mixed.columns)} cols)")
    print(f"[mixed] Written: {test_out}   ({len(test_mixed)} rows, {len(test_mixed.columns)} cols)")
    print(f"[mixed] Columns: {list(train_mixed.columns)}")


if __name__ == "__main__":
    main()