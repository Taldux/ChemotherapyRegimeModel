"""
@file synth_gen.py
@brief Generate a fully synthetic dataset in the schema expected by src.train.

Outputs:
    <out_dir>/train.csv
    <out_dir>/test.csv
    <out_dir>/test_mixed.csv
    <out_dir>/meta.json

Usage:
    uv run python -m src.preprocess.synth_gen --n_train 1000 --n_test 200
    uv run python -m src.preprocess.synth_gen --n_patients 500
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


try:
    from src.preprocess.preprocess import SUBSCALE_COLS, TARGET_COL, TREATMENT_COL
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.preprocess.preprocess import SUBSCALE_COLS, TARGET_COL, TREATMENT_COL


OUTPUT_DIR = "data/synthetic_preprocessed"
RANDOM_SEED = 42
TREATMENTS = [0, 1, 2]
TREATMENT_NAMES = {
    0: "Cisplatin/Navelbine",
    1: "Docetaxel",
    2: "Gemcitabine",
}
TREATMENT_PROBS = np.array([0.46, 0.18, 0.36], dtype=float)
N_DIAGNOSES = 25

SUBSCALE_STATS = {
    "EORTC_QLQ_C30__3_1__SUBSCALE_Physical_Functioning": (64.9, 23.6),
    "EORTC_QLQ_C30__3_1__SUBSCALE_Role_Functioning": (50.4, 28.8),
    "EORTC_QLQ_C30__3_1__SUBSCALE_Emotional_Functioning": (68.2, 24.1),
    "EORTC_QLQ_C30__3_1__SUBSCALE_Cognitive_Functioning": (83.2, 21.0),
    "EORTC_QLQ_C30__3_1__SUBSCALE_Global_Quality_of_Life": (53.2, 18.2),
    "EORTC_QLQ_C30__3_1__SUBSCALE_Fatigue": (48.9, 25.3),
    "EORTC_QLQ_C30__3_1__SUBSCALE_Nausea___Vomiting": (11.7, 21.1),
    "EORTC_QLQ_C30__3_1__SUBSCALE_Pain": (30.9, 29.3),
    "EORTC_QLQ_C30__3_1__SUBSCALE_Sleep_Disturbances": (30.9, 31.6),
    "EORTC_QLQ_C30__3_1__SUBSCALE_Appetite_Loss": (27.5, 33.0),
}


def _clip_score(value: float) -> float:
    """@brief Clip an EORTC score to the valid range [1, 100]."""
    return float(np.clip(value, 1.0, 100.0))


def _softmax(logits: np.ndarray) -> np.ndarray:
    """
    @brief Numerically stable softmax over a 1-D logit array.

    @param logits  Raw logit array.
    @return Probability array summing to 1.
    """
    logits = np.asarray(logits, dtype=float)
    logits = logits - logits.max()
    probs = np.exp(logits)
    return probs / probs.sum()


def _sample_baseline(rng: np.random.Generator) -> dict[str, float]:
    """
    @brief Sample baseline EORTC subscale values from their empirical distributions.

    @param rng  NumPy random generator.
    @return Dict mapping subscale column name to clipped score.
    """
    return {
        col: _clip_score(rng.normal(*SUBSCALE_STATS[col]))
        for col in SUBSCALE_COLS
    }


def _treatment_assignment_probabilities(baseline: dict[str, float], rng: np.random.Generator) -> np.ndarray:
    """
    @brief Compute covariate-informed treatment assignment probabilities.

    @param baseline  Dict of baseline EORTC subscale values.
    @param rng       NumPy random generator.
    @return Probability array of shape (n_treatments,).
    """
    phys = baseline["EORTC_QLQ_C30__3_1__SUBSCALE_Physical_Functioning"] / 100.0
    role = baseline["EORTC_QLQ_C30__3_1__SUBSCALE_Role_Functioning"] / 100.0
    emotional = baseline["EORTC_QLQ_C30__3_1__SUBSCALE_Emotional_Functioning"] / 100.0
    fatigue = baseline["EORTC_QLQ_C30__3_1__SUBSCALE_Fatigue"] / 100.0
    pain = baseline["EORTC_QLQ_C30__3_1__SUBSCALE_Pain"] / 100.0

    logits = np.array([
        1.0 * phys - 1.2 * fatigue + 0.2 * rng.normal(),
        0.8 * (1.0 - pain) + 0.3 * emotional + 0.2 * rng.normal(),
        0.7 * (1.0 - fatigue) + 0.4 * role + 0.2 * rng.normal(),
    ])
    logits = logits + np.log(TREATMENT_PROBS)
    return _softmax(logits)


def _next_treatment(
    current_treatment: int,
    covariates: dict[str, float],
    potential_outcomes: dict[int, float],
    visit_nr: int,
    rng: np.random.Generator,
) -> int:
    """
    @brief Sample the treatment for the next visit, allowing state-dependent switching.

    Switch probability = clip(0.03 + 0.0015*regret + 0.03*visit_nr, 0.02, 0.35).
    When a switch occurs, the new arm is sampled from a utility-weighted softmax
    with a penalty on returning to the current arm.

    @param current_treatment   Treatment index used in this visit.
    @param covariates          Current visit EORTC subscale values.
    @param potential_outcomes  Dict mapping treatment index to potential outcome.
    @param visit_nr            Zero-based visit number.
    @param rng                 NumPy random generator.
    @return Treatment index for the next visit.
    """
    base_probs = _treatment_assignment_probabilities(covariates, rng)
    utilities = np.array([potential_outcomes[t] for t in TREATMENTS], dtype=float)
    current_utility = potential_outcomes[current_treatment]
    regret = max(0.0, float(utilities.max() - current_utility))

    # Later visits and large expected regret increase the chance of switching,
    # while still keeping most trajectories fairly persistent.
    switch_prob = float(np.clip(0.03 + 0.0015 * regret + 0.03 * visit_nr, 0.02, 0.35))
    if rng.random() >= switch_prob:
        return current_treatment

    alt_logits = np.log(base_probs + 1e-9) + (utilities - utilities.mean()) / 45.0
    alt_logits[current_treatment] -= 4.0
    alt_probs = _softmax(alt_logits)
    return int(rng.choice(TREATMENTS, p=alt_probs))


def _visit_covariates(
    baseline: dict[str, float],
    visit_nr: int,
    patient_trend: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """
    @brief Generate EORTC subscale values for one visit given patient trajectory.

    @param baseline       Baseline subscale values.
    @param visit_nr       Zero-based visit number.
    @param patient_trend  Per-patient drift rate (positive = deteriorating).
    @param rng            NumPy random generator.
    @return Dict mapping subscale column name to visit-level clipped score.
    """
    values = {}
    for col in SUBSCALE_COLS:
        base = baseline[col]
        is_function = "Functioning" in col or "Quality_of_Life" in col
        direction = -1.0 if is_function else 1.0
        drift = direction * patient_trend * visit_nr
        values[col] = _clip_score(base + drift + rng.normal(0.0, 4.0))
    return values


def _potential_outcomes(
    covariates: dict[str, float],
    remaining_days: float,
    diagnosis_enc: int,
    sex_enc: int,
    rng: np.random.Generator,
) -> dict[int, float]:
    """
    @brief Compute potential outcomes for all treatment arms at one visit.

    @param covariates      Current visit EORTC subscale values.
    @param remaining_days  Days between this visit and final follow-up.
    @param diagnosis_enc   Encoded diagnosis category.
    @param sex_enc         Encoded sex (0 or 1).
    @param rng             NumPy random generator.
    @return Dict mapping treatment index to simulated time-to-last-days outcome.
    """
    phys = covariates["EORTC_QLQ_C30__3_1__SUBSCALE_Physical_Functioning"] / 100.0
    role = covariates["EORTC_QLQ_C30__3_1__SUBSCALE_Role_Functioning"] / 100.0
    emotional = covariates["EORTC_QLQ_C30__3_1__SUBSCALE_Emotional_Functioning"] / 100.0
    qol = covariates["EORTC_QLQ_C30__3_1__SUBSCALE_Global_Quality_of_Life"] / 100.0
    fatigue = covariates["EORTC_QLQ_C30__3_1__SUBSCALE_Fatigue"] / 100.0
    pain = covariates["EORTC_QLQ_C30__3_1__SUBSCALE_Pain"] / 100.0
    sleep = covariates["EORTC_QLQ_C30__3_1__SUBSCALE_Sleep_Disturbances"] / 100.0

    diagnosis_term = (diagnosis_enc % 5 - 2) * 8.0
    sex_term = 6.0 if sex_enc == 0 else -6.0
    base = (
        remaining_days
        + 80.0 * qol
        + 40.0 * phys
        + 20.0 * emotional
        - 60.0 * fatigue
        - 45.0 * pain
        - 20.0 * sleep
        + diagnosis_term
        + sex_term
    )

    treatment_effects = {
        0: 35.0 * phys + 20.0 * qol - 25.0 * fatigue,
        1: 45.0 * (1.0 - pain) + 15.0 * emotional - 10.0 * phys,
        2: 50.0 * (1.0 - fatigue) + 15.0 * role - 20.0 * sleep,
    }
    return {
        treatment: float(np.clip(base + treatment_effects[treatment] + rng.normal(0.0, 10.0), 0.0, 1500.0))
        for treatment in TREATMENTS
    }


def generate_split(
    n_patients: int,
    patient_id_offset: int = 0,
    seed: int = RANDOM_SEED,
    include_counterfactuals: bool = False,
) -> pd.DataFrame:
    """
    @brief Generate a synthetic panel DataFrame for one data split.

    @param n_patients             Number of patients to simulate.
    @param patient_id_offset      Starting patient ID offset (avoids train/test ID collision).
    @param seed                   Random seed.
    @param include_counterfactuals  If True, add Y0, Y1, Y2 potential outcome columns.
    @return DataFrame with one row per patient visit, sorted by [Patienten_ID, visit_nr].
    """
    rng = np.random.default_rng(seed)
    rows = []

    for pid in range(patient_id_offset + 1, patient_id_offset + n_patients + 1):
        baseline = _sample_baseline(rng)
        sex_enc = int(rng.integers(0, 2))
        diagnosis_enc = int(rng.integers(0, N_DIAGNOSES))
        patient_trend = float(rng.normal(1.2, 0.4))
        n_visits = int(rng.integers(3, 11))
        if n_visits == 1:
            n_visits = 2

        intervals = rng.integers(14, 42, size=n_visits - 1)
        visit_days = np.concatenate([[0], np.cumsum(intervals)]).astype(int)
        remaining_buffer = int(rng.integers(30, 180))
        final_day = int(visit_days[-1] + remaining_buffer)
        base_date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(rng.integers(0, 365 * 3)))

        observed_t = int(rng.choice(TREATMENTS, p=_treatment_assignment_probabilities(baseline, rng)))

        for visit_nr, day in enumerate(visit_days):
            covariates = _visit_covariates(baseline, visit_nr, patient_trend, rng)
            remaining_days = float(final_day - day)
            y_values = _potential_outcomes(covariates, remaining_days, diagnosis_enc, sex_enc, rng)

            row = {
                "Patienten_ID": pid,
                "examdate": (base_date + pd.Timedelta(days=int(day))).strftime("%Y-%m-%d"),
                TARGET_COL: int(round(y_values[observed_t])),
                TREATMENT_COL: observed_t,
                "Geschlecht_enc": sex_enc,
                "Initiale_Diagnose_enc": diagnosis_enc,
                "visit_nr": visit_nr,
            }
            row.update(covariates)

            if include_counterfactuals:
                for treatment in TREATMENTS:
                    row[f"Y{treatment}"] = y_values[treatment]

            rows.append(row)

            if visit_nr < len(visit_days) - 1:
                observed_t = _next_treatment(
                    observed_t,
                    covariates,
                    y_values,
                    visit_nr,
                    rng,
                )

    df = pd.DataFrame(rows)
    ordered = [
        "Patienten_ID",
        "examdate",
        TARGET_COL,
        TREATMENT_COL,
        *SUBSCALE_COLS,
        "Geschlecht_enc",
        "Initiale_Diagnose_enc",
        "visit_nr",
    ]
    if include_counterfactuals:
        ordered.extend([f"Y{t}" for t in TREATMENTS])
    return df[ordered].sort_values(["Patienten_ID", "visit_nr"]).reset_index(drop=True)


def save_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str) -> None:
    """
    @brief Write train.csv, test.csv, test_mixed.csv, and meta.json to disk.

    @param train_df  Training split DataFrame (no counterfactual columns).
    @param test_df   Test split DataFrame (includes counterfactual Y-columns).
    @param out_dir   Output directory path; created if it does not exist.
    """
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test_df.drop(columns=[f"Y{t}" for t in TREATMENTS]).to_csv(os.path.join(out_dir, "test.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test_mixed.csv"), index=False)

    meta = {
        "data_source": "synth",
        "feature_cols": [*SUBSCALE_COLS, "Geschlecht_enc", "Initiale_Diagnose_enc"],
        "target_col": TARGET_COL,
        "treatment_col": TREATMENT_COL,
        "treatments": TREATMENT_NAMES,
        "train_patients": int(train_df["Patienten_ID"].nunique()),
        "test_patients": int(test_df["Patienten_ID"].nunique()),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def main() -> None:
    """
    @brief CLI entry point: parse arguments and generate the synthetic dataset.
    """
    parser = argparse.ArgumentParser(description="Generate fully synthetic train/test data for src.train.")
    parser.add_argument("--n_train", type=int, default=None, help="Number of train patients")
    parser.add_argument("--n_test", type=int, default=None, help="Number of test patients")
    parser.add_argument(
        "--n_patients",
        type=int,
        default=None,
        help="Backward-compatible shorthand: sets n_train and n_test when those are omitted.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--out_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.n_patients is not None:
        if args.n_train is None:
            args.n_train = args.n_patients
        if args.n_test is None:
            args.n_test = args.n_patients

    if args.n_train is None:
        args.n_train = 1000
    if args.n_test is None:
        args.n_test = 200

    if args.n_train <= 0 or args.n_test <= 0:
        raise ValueError("n_train and n_test must both be positive integers")

    print(f"[synth] Generating fully synthetic train split with {args.n_train} patients …")
    train_df = generate_split(args.n_train, patient_id_offset=0, seed=args.seed, include_counterfactuals=False)

    print(f"[synth] Generating fully synthetic test split with {args.n_test} patients …")
    test_df = generate_split(
        args.n_test,
        patient_id_offset=args.n_train,
        seed=args.seed + 1,
        include_counterfactuals=True,
    )

    save_dataset(train_df, test_df, args.out_dir)
    print(f"[synth] Written dataset to {args.out_dir}")
    print(f"[synth] train.csv: {len(train_df)} rows, {train_df['Patienten_ID'].nunique()} patients")
    print(f"[synth] test_mixed.csv: {len(test_df)} rows, {test_df['Patienten_ID'].nunique()} patients")


if __name__ == "__main__":
    main()
