"""
@file train.py
@brief Train causal models on preprocessed panel data and evaluate visit-by-visit.

Data flow:
    semi  -> data/preprocessed/train.csv      + data/preprocessed/test_mixed.csv
    synth -> data/synthetic_preprocessed/train.csv + data/synthetic_preprocessed/test_mixed.csv

Evaluation protocol: for each patient and each visit t:
  - LSTM models receive the padded 3-D sequence of visits 0...t
  - Other models receive aggregated history (mean, std, last, prev, n_visits)
  - true_best = argmax_k(Y{k}_true);  pred_best = argmax_k(pred_Y{k})
  - BOWL: recommendation only (no Y predictions)
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from src.plots import (
    plot_comparative_ite_distribution,
    plot_comparative_scatter,
    plot_policy_comparison,
    plot_pehe_comparison,
    plot_ate_error_comparison,
    plot_pehe_heatmap,
)
from src.models.slearner import SLearner
from src.models.tlearner import TLearner
from src.models.bowl import BOWL
from src.models.causal_forest import CausalForest
from src.models.dynamic_dml import DynamicDMLModel
from src.models.lstm_model import LSTMRegressor


DATA_SOURCE_DIRS = {
    "semi": "data/preprocessed",
    "synth": "data/synthetic_preprocessed",
    "crn": "src/crn/preprocessed",
}

TARGET_COL    = "time_to_last_days"
TREATMENT_COL = "Chemotherapieregime_enc"
PATIENT_COL   = "Patienten_ID"
VISIT_COL     = "visit_nr"
DATE_COL      = "examdate"

NON_FEATURE_COLS = {TARGET_COL, TREATMENT_COL, PATIENT_COL, VISIT_COL, DATE_COL}

ALL_MODEL_KEYS = [
    "slearner",
    "tlearner",
    "bowl",
    "causal_forest",
    "dynamic_dml",
    "lstm_s",
    "lstm_t",
]

RF_PARAMS = dict(
    n_estimators=100, max_depth=10, min_samples_split=20,
    min_samples_leaf=10, random_state=42, n_jobs=-1,
)

LSTM_PARAMS = dict(
    hidden_dim=64, num_layers=2, dropout=0.2, lr=1e-3,
    epochs=200, batch_size=64, patience=15, random_state=42,
)

# ── Time-series mode per model ─────────────────────────────────────────────────
# True  → use full visit history at evaluation time:
#           LSTM models receive a padded 3-D sequence.
#           DDML receives the raw sequence and past treatments for trajectory-based prediction.
#           Other models receive aggregated history features
#           (mean, std, last, prev, n_visits) via _aggregate_sequence.
# False → use only the current visit's raw feature row (cross-sectional).
USE_TIMESERIES: dict = {
    "slearner":      True,
    "tlearner":      True,
    "bowl":          True,
    "causal_forest": True,
    "dynamic_dml":   True,
    "lstm_s":        True,
    "lstm_t":        True,
}

TIMESERIES_INCOMPATIBLE: set = {}


def _resolve_data_dir(data_source: str) -> str:
    """
    @brief Resolve data directory path from a data_source key.

    @param data_source  One of DATA_SOURCE_DIRS keys.
    @return Absolute directory path string.
    """
    if data_source not in DATA_SOURCE_DIRS:
        raise ValueError(f"Unknown data source '{data_source}'. Choices: {list(DATA_SOURCE_DIRS)}")
    return DATA_SOURCE_DIRS[data_source]


def _missing_data_hint(data_source: str) -> str:
    """
    @brief Return a user-facing hint when required CSV files are missing.

    @param data_source  Data source key.
    @return Hint string pointing to the correct preprocessing command.
    """
    if data_source == "synth":
        return "Run 'uv run python -m src.preprocess.synth_gen' first."
    return "Run 'uv run python -m src.preprocess.semisynth_gen' first if test_mixed.csv is missing."


def load_panel_data(split: str, data_dir: str, data_source: str) -> pd.DataFrame:
    """
    @brief Load a preprocessed split CSV sorted into panel order.

    @param split        'train' or 'test'.
    @param data_dir     Resolved data directory path.
    @param data_source  Source key used for error hints.
    @return DataFrame with one row per patient visit, sorted by [Patienten_ID, visit_nr].
    """
    path = os.path.join(data_dir, f"{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}. {_missing_data_hint(data_source)}")
    df   = pd.read_csv(path, low_memory=False)
    df   = df.sort_values([PATIENT_COL, VISIT_COL]).reset_index(drop=True)
    print(f"  [{split}] {len(df)} rows  |  {df[PATIENT_COL].nunique()} patients")
    return df


def load_test_mixed(data_dir: str, data_source: str) -> pd.DataFrame:
    """
    @brief Load test_mixed.csv for the selected data source.

    @param data_dir     Resolved data directory path.
    @param data_source  Source key used for error hints.
    @return DataFrame sorted by [Patienten_ID, visit_nr] including Y{k} columns.
    """
    path = os.path.join(data_dir, "test_mixed.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}. {_missing_data_hint(data_source)}")
    df   = pd.read_csv(path, low_memory=False)
    df   = df.sort_values([PATIENT_COL, VISIT_COL]).reset_index(drop=True)
    y_cols = sorted(
        [c for c in df.columns if c.startswith("Y") and c[1:].isdigit()],
        key=lambda c: int(c[1:]),
    )
    if not y_cols:
        raise ValueError(f"No Y-columns found in {path}. {_missing_data_hint(data_source)}")
    print(f"  [test_mixed] {len(df)} rows  |  "
          f"{df[PATIENT_COL].nunique()} patients  |  Y-cols: {y_cols}")
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    """
    @brief Return feature columns, excluding non-feature and Y-outcome columns.

    @param df  Any preprocessed DataFrame.
    @return List of column names to use as model input features.
    """
    return [c for c in df.columns
            if c not in NON_FEATURE_COLS
            and not (c.startswith("Y") and c[1:].isdigit())]


def get_treatment_indices(df: pd.DataFrame) -> list:
    """
    @brief Return sorted unique treatment indices from Y{k} column names.

    @param df  DataFrame with Y0, Y1, ... columns.
    @return Sorted list of treatment indices, e.g. [0, 1, 2].
    """
    return sorted(
        int(c[1:]) for c in df.columns
        if c.startswith("Y") and c[1:].isdigit()
    )


def prepare_panel_arrays(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    @brief Extract numpy arrays for panel fitting from a sorted DataFrame.

    @param df           Panel DataFrame sorted by [Patienten_ID, visit_nr].
    @param feature_cols Feature columns to extract.
    @return Tuple (X, y, T, groups) each of shape (n_rows, ...).
    """
    return (
        df[feature_cols].values.astype(np.float64),
        df[TARGET_COL].values.astype(np.float64),
        df[TREATMENT_COL].values.astype(int),
        df[PATIENT_COL].values,
    )


def _is_lstm(model) -> bool:
    """
    @brief Return True if the model wraps an LSTM base estimator.

    @param model  Any model instance.
    @return True when base_model._is_sequential is True.
    """
    return getattr(getattr(model, "base_model", None), "_is_sequential", False)


def _seq_len(model) -> int:
    """
    @brief Return the sequence length the model was balanced to during fit_panel.

    @param model  Fitted panel model.
    @return seq_len_ attribute if present, otherwise 1.
    """
    return getattr(model, "seq_len_", 1)


def _pad_sequence(features_2d: np.ndarray, seq_len: int) -> np.ndarray:
    """
    @brief Pad or truncate a (n_steps, n_features) array to (1, seq_len, n_features).

    Short sequences are LOCF-padded at the start; long sequences are tail-truncated.

    @param features_2d  Array of shape (n_steps, n_features).
    @param seq_len      Target number of time steps.
    @return Float32 array of shape (1, seq_len, n_features).
    """
    n = len(features_2d)
    if n >= seq_len:
        seq = features_2d[-seq_len:]
    else:
        pad = np.tile(features_2d[0], (seq_len - n, 1))
        seq = np.vstack([pad, features_2d])
    return seq[np.newaxis].astype(np.float32)


def _aggregate_sequence(features_so_far: np.ndarray) -> np.ndarray:
    """
    @brief Summarise a growing visit history into a fixed-width feature vector.

    Computes per-feature (mean, std, last, prev) over all visits seen so far,
    then appends the visit count. Result shape: (1, n_features * 4 + 1).

    @param features_so_far  Array of shape (n_steps, n_features).
    @return Float64 array of shape (1, n_features * 4 + 1).
    """
    mean = features_so_far.mean(axis=0)
    std  = features_so_far.std(axis=0)
    last = features_so_far[-1]
    prev = features_so_far[-2] if len(features_so_far) > 1 else np.zeros_like(last)
    n    = np.array([len(features_so_far)], dtype=np.float64)
    return np.concatenate([mean, std, last, prev, n])[np.newaxis]


def _build_aggregated_panel(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    @brief Convert a raw panel DataFrame into per-step aggregated feature rows.

    For each patient and visit step t, produces the same vector that
    _aggregate_sequence would produce at evaluation time.

    @param df           Panel DataFrame sorted by [Patienten_ID, visit_nr].
    @param feature_cols Raw feature column names.
    @return Tuple (X_agg, y, T, groups) where X_agg has shape (n_total_steps, n_agg_features).
    """
    X_rows, y_rows, T_rows, g_rows = [], [], [], []

    for pid, pat in df.groupby(PATIENT_COL, sort=False):
        pat     = pat.sort_values(VISIT_COL)
        feats   = pat[feature_cols].values.astype(np.float64)
        targets = pat[TARGET_COL].values.astype(np.float64)
        treats  = pat[TREATMENT_COL].values.astype(int)

        for step_idx in range(len(pat)):
            agg = _aggregate_sequence(feats[: step_idx + 1])
            X_rows.append(agg[0])
            y_rows.append(targets[step_idx])
            T_rows.append(treats[step_idx])
            g_rows.append(pid)

    return (
        np.array(X_rows, dtype=np.float64),
        np.array(y_rows,  dtype=np.float64),
        np.array(T_rows,  dtype=int),
        np.array(g_rows),
    )


def _predict_outcomes_at_step(mkey: str, model, features_so_far: np.ndarray,
                               treatments: list, past_treatments: np.ndarray) -> dict:
    """
    @brief Predict the outcome under every treatment at one visit step.

    Routing: dynamic_dml -> trajectory-based panel prediction using
    past_treatments; LSTM with USE_TIMESERIES -> padded 3-D sequence;
    other with USE_TIMESERIES -> aggregated history vector; else -> raw
    current row.

    @param mkey            Model key for USE_TIMESERIES lookup.
    @param model           Fitted model with predict_outcome method.
    @param features_so_far Array of shape (n_steps_so_far, n_features).
    @param treatments      Treatment indices to predict for.
    @param past_treatments Array of shape (n_steps_so_far - 1,) with the
                           treatments given before the current visit.
    @return Dict {treatment_index: predicted_outcome_scalar}.
    """
    if mkey == "dynamic_dml":
        return {
            t: float(model.predict_outcome_sequential(features_so_far,
                                                      past_treatments, t))
            for t in treatments
        }

    if USE_TIMESERIES.get(mkey, False) and _is_lstm(model):
        sl   = _seq_len(model)
        X_3d = _pad_sequence(features_so_far, sl)
        model.set_eval_sequences(X_3d)
        X_input = features_so_far[-1:].astype(np.float64)
    elif USE_TIMESERIES.get(mkey, False):
        X_input = _aggregate_sequence(features_so_far)
    else:
        X_input = features_so_far[-1:].astype(np.float64)

    return {t: float(model.predict_outcome(X_input, t)[0]) for t in treatments}


def _build_models(active: set) -> dict:
    """
    @brief Instantiate all requested models keyed by model key.

    @param active  Subset of ALL_MODEL_KEYS to instantiate.
    @return Dict {model_key: unfitted_model_or_None}.
    """
    rf        = RandomForestRegressor(**RF_PARAMS)
    lr        = LogisticRegression(C=1.0, max_iter=1000)
    lstm_base = (LSTMRegressor(**LSTM_PARAMS)
                 if ("lstm_s" in active or "lstm_t" in active) else None)
    return {
        "slearner":      SLearner(base_model=rf)                              if "slearner"      in active else None,
        "tlearner":      TLearner(base_model=rf)                              if "tlearner"      in active else None,
        "bowl":          BOWL(base_classifier=lr, scale_outcomes=True)        if "bowl"          in active else None,
        "causal_forest": CausalForest(n_estimators=200, min_samples_leaf=10,
                                       random_state=42)                       if "causal_forest" in active else None,
        "dynamic_dml":   DynamicDMLModel(n_estimators=100, max_depth=10,
                                          min_samples_leaf=5, cv=2,
                                          random_state=42)                    if "dynamic_dml"   in active else None,
        "lstm_s":        SLearner(base_model=lstm_base)                       if "lstm_s"        in active else None,
        "lstm_t":        TLearner(base_model=lstm_base)                       if "lstm_t"        in active else None,
    }


def train_models(models: dict, X: np.ndarray, y: np.ndarray,
                 T: np.ndarray, groups: np.ndarray,
                 X_agg: np.ndarray, y_agg: np.ndarray,
                 T_agg: np.ndarray, groups_agg: np.ndarray) -> None:
    """
    @brief Fit every non-None model in-place on the training panel.

    USE_TIMESERIES=True models (non-LSTM, non-incompatible) are trained on
    X_agg so training and evaluation use identical features. LSTM models use
    raw panel via fit_panel. TIMESERIES_INCOMPATIBLE models always use raw X.

    @param models      Model key -> unfitted model instance or None.
    @param X           Raw feature matrix (n_rows, n_features).
    @param y           Target vector (n_rows,).
    @param T           Treatment vector (n_rows,).
    @param groups      Patient IDs (n_rows,).
    @param X_agg       Aggregated feature matrix (n_rows, n_agg_features).
    @param y_agg       Target vector aligned with aggregated order.
    @param T_agg       Treatment vector aligned with aggregated order.
    @param groups_agg  Patient IDs aligned with aggregated order.
    """
    def _use_agg(mkey: str) -> bool:
        return (
            USE_TIMESERIES.get(mkey, False)
            and not _is_lstm(models[mkey])
            and mkey not in TIMESERIES_INCOMPATIBLE
        )

    def _X(mkey):  return X_agg      if _use_agg(mkey) else X
    def _y(mkey):  return y_agg      if _use_agg(mkey) else y
    def _T(mkey):  return T_agg      if _use_agg(mkey) else T
    def _G(mkey):  return groups_agg if _use_agg(mkey) else groups

    if models.get("slearner"):
        print("  Training S-Learner (Random Forest)…")
        models["slearner"].fit_panel(_y("slearner"), _T("slearner"),
                                     _X("slearner"), _G("slearner"))

    if models.get("tlearner"):
        print("  Training T-Learner (Random Forest)…")
        models["tlearner"].fit_panel(_y("tlearner"), _T("tlearner"),
                                     _X("tlearner"), _G("tlearner"))

    if models.get("bowl"):
        print("  Training BOWL (Logistic Regression)…")
        models["bowl"].fit_panel(Y=_y("bowl"), T=_T("bowl"),
                                 X=_X("bowl"), groups=_G("bowl"))

    if models.get("causal_forest"):
        print("  Training Causal Forest (econml CausalForestDML)…")
        models["causal_forest"].fit_panel(Y=_y("causal_forest"), T=_T("causal_forest"),
                                          X=_X("causal_forest"), groups=_G("causal_forest"))

    if models.get("dynamic_dml"):
        print("  Training DynamicDML (panel path, raw per-visit features)…")
        models["dynamic_dml"].fit_panel(y, T, X, groups)
        print(f"    DynamicDML fit mode: {models['dynamic_dml'].fit_mode_}")

    if models.get("lstm_s"):
        print("  Training LSTM S-Learner…")
        models["lstm_s"].fit_panel(y, T, X, groups)

    if models.get("lstm_t"):
        print("  Training LSTM T-Learner…")
        models["lstm_t"].fit_panel(y, T, X, groups)


def _evaluate_patient(pid, pat_df: pd.DataFrame,
                       feature_cols: list, treatments: list,
                       outcome_models: dict, bowl_model) -> list:
    """
    @brief Evaluate all models for one patient across every visit step.

    At each step, the growing feature window (visits 0...t) and the past
    treatment trajectory (visits 0...t-1) are passed to the predictors.
    true_best = argmax_k(Y{k}_true).

    @param pid            Patient identifier.
    @param pat_df         All rows for this patient sorted by visit_nr.
    @param feature_cols   Input feature column names.
    @param treatments     Treatment indices, e.g. [0, 1, 2].
    @param outcome_models Dict {model_key: fitted_model} excluding BOWL.
    @param bowl_model     Fitted BOWL instance or None.
    @return List of dicts, one per visit step.
    """
    features_all     = pat_df[feature_cols].values.astype(np.float64)
    treatments_all   = pat_df[TREATMENT_COL].values.astype(int)
    records          = []

    for step_idx, (_, row) in enumerate(pat_df.iterrows()):
        features_so_far  = features_all[: step_idx + 1]
        past_treatments  = treatments_all[: step_idx]

        real_treatment = int(row[TREATMENT_COL])
        real_y         = float(row[TARGET_COL])
        y_true         = {k: float(row[f"Y{k}"]) for k in treatments}
        true_best      = max(treatments, key=lambda k: y_true[k])

        record = {
            PATIENT_COL:      pid,
            VISIT_COL:        int(row[VISIT_COL]),
            "real_treatment": real_treatment,
            "real_y":         real_y,
            "true_best":      true_best,
        }
        for k in treatments:
            record[f"Y{k}_true"] = y_true[k]

        for mkey, model in outcome_models.items():
            preds = _predict_outcomes_at_step(
                mkey, model, features_so_far, treatments, past_treatments,
            )
            pred_best = max(treatments, key=lambda k: preds[k])
            for k in treatments:
                record[f"{mkey}_pred_Y{k}"] = preds[k]
            record[f"{mkey}_best"]    = pred_best
            record[f"{mkey}_correct"] = int(pred_best == true_best)

        if bowl_model is not None:
            if USE_TIMESERIES.get("bowl", False):
                X_bowl = _aggregate_sequence(features_so_far)
            else:
                X_bowl = features_so_far[-1:].astype(np.float64)
            recommended = int(bowl_model.predict_optimal_treatment(X_bowl)[0])
            record["bowl_best"]    = recommended
            record["bowl_correct"] = int(recommended == true_best)

        records.append(record)

    return records


def evaluate(models: dict, test_mixed_df: pd.DataFrame,
             feature_cols: list, treatments: list) -> pd.DataFrame:
    """
    @brief Run visit-by-visit evaluation across all test patients.

    @param models         Model key -> fitted model or None.
    @param test_mixed_df  Sorted test_mixed.csv DataFrame.
    @param feature_cols   Input feature column names.
    @param treatments     Sorted treatment indices.
    @return DataFrame with one row per (patient, visit).
    """
    outcome_models = {k: m for k, m in models.items()
                      if m is not None and k != "bowl"}
    bowl_model     = models.get("bowl")

    patients    = sorted(test_mixed_df[PATIENT_COL].unique())
    all_records = []
    n           = len(patients)

    for i, pid in enumerate(patients):
        if (i + 1) % max(1, n // 10) == 0 or i == 0:
            print(f"  Patient {i + 1:>4}/{n}…")

        pat_df = (test_mixed_df[test_mixed_df[PATIENT_COL] == pid]
                  .sort_values(VISIT_COL)
                  .reset_index(drop=True))

        all_records.extend(
            _evaluate_patient(pid, pat_df, feature_cols, treatments,
                              outcome_models, bowl_model)
        )

    return pd.DataFrame(all_records).reset_index(drop=True)


def compute_pehe(results: pd.DataFrame, outcome_keys: list,
                 treatments: list) -> dict:
    """
    @brief Compute PEHE per model per treatment pair.

    PEHE = sqrt(mean((tau_pred - tau_true)^2))

    @param results       Output of evaluate().
    @param outcome_keys  Model keys that produce Y predictions (no BOWL).
    @param treatments    Treatment indices.
    @return Dict {pair_label: {model_key: pehe_float}}.
    """
    pehe_data = {}
    for from_t, to_t in itertools.combinations(treatments, 2):
        label    = f"T{from_t} vs T{to_t}"
        tau_true = (results[f"Y{to_t}_true"].values
                    - results[f"Y{from_t}_true"].values)
        for mkey in outcome_keys:
            tau_pred = (results[f"{mkey}_pred_Y{to_t}"].values
                        - results[f"{mkey}_pred_Y{from_t}"].values)
            pehe_data.setdefault(label, {})[mkey] = float(
                np.sqrt(np.mean((tau_pred - tau_true) ** 2))
            )
    return pehe_data


def compute_ate_error(results: pd.DataFrame, outcome_keys: list,
                      treatments: list) -> dict:
    """
    @brief Compute ATE error per model per treatment pair.

    ATE error = |mean(tau_pred) - mean(tau_true)|

    @param results       Output of evaluate().
    @param outcome_keys  Model keys that produce Y predictions (no BOWL).
    @param treatments    Treatment indices.
    @return Dict {pair_label: {model_key: ate_error_float}}.
    """
    ate_data = {}
    for from_t, to_t in itertools.combinations(treatments, 2):
        label    = f"T{from_t} vs T{to_t}"
        tau_true = (results[f"Y{to_t}_true"].values
                    - results[f"Y{from_t}_true"].values)
        for mkey in outcome_keys:
            tau_pred = (results[f"{mkey}_pred_Y{to_t}"].values
                        - results[f"{mkey}_pred_Y{from_t}"].values)
            ate_data.setdefault(label, {})[mkey] = float(
                abs(tau_pred.mean() - tau_true.mean())
            )
    return ate_data


def compute_accuracy(results: pd.DataFrame, all_active_keys: list) -> dict:
    """
    @brief Compute treatment recommendation accuracy per model.

    Accuracy = fraction of steps where pred_best == true_best.

    @param results          Output of evaluate().
    @param all_active_keys  All active model keys including BOWL.
    @return Dict {model_key: accuracy_float in [0, 1]}.
    """
    return {
        mkey: float(results[f"{mkey}_correct"].mean())
        for mkey in all_active_keys
        if f"{mkey}_correct" in results.columns
    }


def compute_policy_value(results: pd.DataFrame, outcome_keys: list,
                         treatments: list) -> dict:
    """
    @brief Compute the expected realised outcome under each model's policy.

    For each step the model picks argmax_k(pred_Y{k}); realised outcome is
    Y{picked_k}_true.

    @param results       Output of evaluate().
    @param outcome_keys  Model keys that produce Y predictions (no BOWL).
    @param treatments    Treatment indices.
    @return Dict {model_key: mean_realised_outcome}.
    """
    val = {}
    for mkey in outcome_keys:
        best_col = f"{mkey}_best"
        if best_col not in results.columns:
            continue
        picked_y = [
            float(results.iloc[i][f"Y{int(results.iloc[i][best_col])}_true"])
            for i in range(len(results))
        ]
        val[mkey] = float(np.mean(picked_y))
    return val


def compute_ci_width(models: dict, test_mixed_df: pd.DataFrame,
                     feature_cols: list, treatments: list,
                     n_mc_samples: int = 50,
                     n_bootstrap: int = 30,
                     bootstrap_lstm: bool = False,
                     alpha: float = 0.05,
                     ci_method: str = "bootstrap",
                     random_state: int = 42) -> tuple[dict, dict]:
    """
    @brief Compute mean 95 % CI width per model per treatment pair.

    Builds the appropriate test feature representation for each model type
    (aggregated 2-D for RF / Causal Forest; padded 3-D for LSTM) and calls
    predict_effect_with_ci() when available.

    @param models          Model key -> fitted model or None.
    @param test_mixed_df   Sorted test_mixed.csv DataFrame.
    @param feature_cols    Input feature column names.
    @param treatments      Treatment indices.
    @param n_mc_samples    MC Dropout passes for fallback LSTM CI.
    @param n_bootstrap     Bootstrap refits for S/T learners.
    @param bootstrap_lstm  If True, allow bootstrap CI for LSTM S/T learners.
    @param alpha           CI significance level (default 0.05).
    @param ci_method       CI method passed to S/T learners.
    @param random_state    Bootstrap RNG seed.
    @return Tuple (
            ci_width_data   : {pair_label: {model_key: mean_ci_width}},
            ci_effect_data  : {pair_label: {model_key: mean_effect_center}}
        ).
    """
    outcome_keys = [k for k in models
                    if models[k] is not None
                    and k != "bowl"
                    and hasattr(models[k], "predict_effect_with_ci")]

    # ── Build test feature representations ────────────────────────────────────
    # 2-D aggregated (for RF / Causal Forest)
    X_agg, *_ = _build_aggregated_panel(test_mixed_df, feature_cols)
    # 2-D raw current-visit features (for DynamicDML effect inference)
    X_raw = test_mixed_df[feature_cols].values.astype(np.float64)

    # 3-D padded sequences (for LSTM models)
    X_seq_map: dict[str, np.ndarray] = {}
    for mkey in outcome_keys:
        model = models[mkey]
        if _is_lstm(model):
            sl = _seq_len(model)
            seqs = []
            for pid, pat in test_mixed_df.groupby(PATIENT_COL, sort=False):
                pat   = pat.sort_values(VISIT_COL)
                feats = pat[feature_cols].values.astype(np.float32)
                for step_idx in range(len(pat)):
                    seqs.append(
                        _pad_sequence(feats[: step_idx + 1], sl)[0]
                    )
            X_seq_map[mkey] = np.stack(seqs)  # (n_steps, seq_len, n_feat)

    # ── Compute CIs ───────────────────────────────────────────────────────────
    ci_data: dict = {}
    ci_effect_data: dict = {}
    for from_t, to_t in itertools.combinations(treatments, 2):
        label = f"T{from_t} vs T{to_t}"
        for mkey in outcome_keys:
            model = models[mkey]
            try:
                if mkey == "dynamic_dml":
                    X_in = X_raw
                elif _is_lstm(model):
                    X_in = X_seq_map[mkey]
                else:
                    X_in = X_agg

                kwargs: dict[str, object] = {"alpha": alpha}
                if model.__class__.__name__ in {"SLearner", "TLearner"}:
                    if _is_lstm(model):
                        if bootstrap_lstm:
                            kwargs.update({
                                "ci_method": ci_method,
                                "n_bootstrap": n_bootstrap,
                                "n_mc_samples": n_mc_samples,
                                "random_state": random_state,
                            })
                        else:
                            kwargs.update({
                                "ci_method": "heuristic",
                                "n_mc_samples": n_mc_samples,
                                "random_state": random_state,
                            })
                    else:
                        kwargs.update({
                            "ci_method": ci_method,
                            "n_bootstrap": n_bootstrap,
                            "n_mc_samples": n_mc_samples,
                            "random_state": random_state,
                        })

                res = model.predict_effect_with_ci(X_in, from_t, to_t, **kwargs)
                width = float(np.mean(res["ci_upper"] - res["ci_lower"]))
                effect_center = float(np.mean(res["effect"]))
                ci_data.setdefault(label, {})[mkey] = width
                ci_effect_data.setdefault(label, {})[mkey] = effect_center
            except Exception:
                pass

    return ci_data, ci_effect_data


def print_summary(results: pd.DataFrame, pehe_data: dict, ate_data: dict,
                  acc_data: dict, val_data: dict, treatments: list,
                  ci_data: dict | None = None,
                  ci_effect_data: dict | None = None) -> None:
    """
    @brief Print a formatted evaluation summary to stdout.

    @param results     Output of evaluate().
    @param pehe_data   Output of compute_pehe().
    @param ate_data    Output of compute_ate_error().
    @param acc_data    Output of compute_accuracy().
    @param val_data    Output of compute_policy_value().
    @param treatments  Treatment indices.
    @param ci_data     Output of compute_ci_width() or None.
    @param ci_effect_data  Mean CI-center effect data or None.
    """
    n_steps    = len(results)
    n_patients = results[PATIENT_COL].nunique()
    print(f"\n  Evaluation steps : {n_steps}  "
          f"({n_patients} patients, {n_steps / n_patients:.1f} visits/patient avg)\n")

    print("─" * 112)
    print("PEHE, ATE error, mean effect center, and mean 95% CI width per treatment pair")
    print("─" * 112)
    for label, model_vals in pehe_data.items():
        for mkey, pehe in model_vals.items():
            ate = ate_data.get(label, {}).get(mkey, float("nan"))
            ciw = ci_data.get(label, {}).get(mkey, float("nan")) if ci_data else float("nan")
            eff = (ci_effect_data.get(label, {}).get(mkey, float("nan"))
                   if ci_effect_data else float("nan"))
            cistr = f"{ciw:>8.4f}" if not np.isnan(ciw) else f"{'--':>8}"
            effstr = f"{eff:>8.4f}" if not np.isnan(eff) else f"{'--':>8}"
            print(
                f"  {label:<22}  {mkey:<20}  PEHE={pehe:>8.4f}"
                f"  ATE err={ate:>8.4f}  Effect={effstr}  CI width={cistr}"
            )

    print("\n" + "─" * 72)
    print(f"  {'Model':<22} {'Accuracy':>12} {'Policy Value':>14}")
    print("─" * 52)
    for mkey in sorted(acc_data):
        acc   = acc_data[mkey]
        pv    = val_data.get(mkey, float("nan"))
        pvstr = f"{pv:>14.2f}" if not np.isnan(pv) else f"{'-':>14}"
        print(f"  {mkey:<22} {acc:>11.1%} {pvstr}")

    oracle_y = [
        float(results.iloc[i][f"Y{int(results.iloc[i]['true_best'])}_true"])
        for i in range(n_steps)
    ]
    rng    = np.random.default_rng(42)
    rand_y = [
        float(results.iloc[i][f"Y{rng.choice(treatments)}_true"])
        for i in range(n_steps)
    ]
    print(f"\n  {'oracle':<22} {'-':>12} {np.mean(oracle_y):>14.2f}")
    print(f"  {'random':<22} {'-':>12} {np.mean(rand_y):>14.2f}")


def generate_plots(results: pd.DataFrame, models: dict,
                   pehe_data: dict, ate_data: dict,
                   acc_data: dict, val_data: dict,
                   treatments: list) -> None:
    """
    @brief Generate and save all evaluation plots to plots/.

    Plots produced: pehe_comparison.png, ate_error_comparison.png,
    pehe_heatmap.png, ite_dist_T{a}_T{b}.png, scatter_T{a}_T{b}.png,
    policy_comparison.png.

    @param results     Output of evaluate().
    @param models      Model key -> fitted model or None.
    @param pehe_data   Output of compute_pehe().
    @param ate_data    Output of compute_ate_error().
    @param acc_data    Output of compute_accuracy().
    @param val_data    Output of compute_policy_value().
    @param treatments  Treatment indices.
    """
    outcome_keys = [k for k in models if models[k] is not None and k != "bowl"]
    pair_labels  = list(pehe_data.keys())

    plot_pehe_comparison(pehe_data, pair_labels, filename="pehe_comparison.png")
    plot_ate_error_comparison(ate_data, pair_labels, filename="ate_error_comparison.png")

    n_t       = len(treatments)
    t_to_idx  = {t: i for i, t in enumerate(treatments)}
    pehe_mats = {k: np.full((n_t, n_t), np.nan) for k in outcome_keys}
    for from_t, to_t in itertools.combinations(treatments, 2):
        label = f"T{from_t} vs T{to_t}"
        i, j  = t_to_idx[from_t], t_to_idx[to_t]
        for mkey in outcome_keys:
            if mkey in pehe_data.get(label, {}):
                pehe_mats[mkey][i, j] = pehe_data[label][mkey]
    plot_pehe_heatmap(pehe_mats,
                      [f"Treatment {t}" for t in treatments],
                      filename="pehe_heatmap.png")

    for from_t, to_t in itertools.combinations(treatments, 2):
        tau_true  = (results[f"Y{to_t}_true"].values
                     - results[f"Y{from_t}_true"].values)
        tau_preds = {
            mkey: (results[f"{mkey}_pred_Y{to_t}"].values
                   - results[f"{mkey}_pred_Y{from_t}"].values)
            for mkey in outcome_keys
            if f"{mkey}_pred_Y{to_t}" in results.columns
        }
        label = f"T{from_t}_to_T{to_t}"
        plot_comparative_ite_distribution(
            tau_true, tau_preds,
            title=f"ITE Distribution  T{from_t} → T{to_t}",
            filename=f"ite_dist_{label}.png",
        )
        plot_comparative_scatter(
            tau_true, tau_preds,
            title=f"Prediction Accuracy  T{from_t} → T{to_t}",
            filename=f"scatter_{label}.png",
        )

    current_y = float(np.mean([
        float(results.iloc[i][f"Y{int(results.iloc[i]['real_treatment'])}_true"])
        for i in range(len(results))
    ]))
    oracle_y = float(np.mean([
        float(results.iloc[i][f"Y{int(results.iloc[i]['true_best'])}_true"])
        for i in range(len(results))
    ]))
    plot_policy_comparison(
        current=current_y,
        oracle=oracle_y,
        models=val_data,
        filename="policy_comparison.png",
    )

    print("All plots written to plots/")


def main(active_keys: list | None = None, data_source: str = "semi") -> None:
    """
    @brief Full pipeline: load data -> train models -> evaluate -> generate plots.

    @param active_keys  Model keys to run; None activates all in ALL_MODEL_KEYS.
    @param data_source  One of DATA_SOURCE_DIRS keys (default 'semi').
    """
    active = set(active_keys if active_keys else ALL_MODEL_KEYS)
    data_dir = _resolve_data_dir(data_source)

    print(f"DATA SOURCE [{data_source}]\n")

    train_df      = load_panel_data("train", data_dir, data_source)
    test_mixed_df = load_test_mixed(data_dir, data_source)

    feature_cols = get_feature_cols(train_df)
    treatments   = get_treatment_indices(test_mixed_df)
    print(f"  Features   ({len(feature_cols)}): {feature_cols}")
    print(f"  Treatments : {treatments}")

    X_tr, y_tr, T_tr, G_tr = prepare_panel_arrays(train_df, feature_cols)

    print("TRAINING MODELS\n")

    print("  Building aggregated panel for non-LSTM timeseries models…")
    X_agg, y_agg, T_agg, G_agg = _build_aggregated_panel(train_df, feature_cols)
    print(f"  Aggregated panel: {X_agg.shape}  (raw: {X_tr.shape})")

    models = _build_models(active)
    train_models(models, X_tr, y_tr, T_tr, G_tr,
                 X_agg, y_agg, T_agg, G_agg)
    print("All models trained.")

    print("EVALUATION\n")

    results = evaluate(models, test_mixed_df, feature_cols, treatments)

    out_path = os.path.join(data_dir, "eval_results.csv")
    results.to_csv(out_path, index=False)
    print(f"  Raw results → {out_path}  ({len(results)} rows)")

    outcome_keys = [k for k in models if models[k] is not None and k != "bowl"]
    all_keys     = [k for k in models if models[k] is not None]

    pehe_data = compute_pehe(results, outcome_keys, treatments)
    ate_data  = compute_ate_error(results, outcome_keys, treatments)
    acc_data  = compute_accuracy(results, all_keys)
    val_data  = compute_policy_value(results, outcome_keys, treatments)

    print("UNCERTAINTY\n")
    ci_data, ci_effect_data = compute_ci_width(
        models, test_mixed_df, feature_cols, treatments
    )

    print_summary(
        results, pehe_data, ate_data, acc_data, val_data,
        treatments, ci_data, ci_effect_data
    )

    print("PLOTS\n")
    generate_plots(results, models, pehe_data, ate_data, acc_data, val_data, treatments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train causal models and evaluate visit-by-visit on test_mixed."
    )
    parser.add_argument(
        "--data", choices=list(DATA_SOURCE_DIRS), default="semi",
        help=f"Data source to use. Choices: {list(DATA_SOURCE_DIRS)}. Default: semi.",
    )
    parser.add_argument(
        "--models", nargs="+", choices=ALL_MODEL_KEYS, default=None,
        help=f"Models to train. Choices: {ALL_MODEL_KEYS}. Default: all.",
    )
    args = parser.parse_args()
    main(active_keys=args.models, data_source=args.data)