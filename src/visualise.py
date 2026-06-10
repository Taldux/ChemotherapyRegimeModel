"""
@file visualise.py
@brief Exploratory analysis of the chemotherapy QoL dataset.

Produces console summaries and figures describing the patient cohort,
chemotherapy regimens, EORTC subscale distributions, and visit cadence.

Default inputs:
  data/raw/c30_test_mypatientreport.csv
  data/preprocessed/train.csv,       data/preprocessed/test.csv
  data/preprocessed/train_mixed.csv, data/preprocessed/test_mixed.csv

Figures are written to plots/data_summary/ by default.
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RAW_PATH = Path("data/raw/c30_test_mypatientreport.csv")
PREP_DIR = Path("data/preprocessed")
FIG_DIR  = Path("plots/data_summary")


ID_COL          = "Patienten_ID"
TREATMENT_COL   = "Chemotherapieregime"
GENDER_COL      = "Geschlecht"
DIAGNOSIS_COL   = "Initiale_Diagnose"
DECEASED_COL    = "Verstorben"
DEATH_DATE_COL  = "Sterbedatum"
EXAM_DATE_COL   = "examdate"

ITEM_PREFIX     = "EORTC_QLQ_C30__3_1__ITEM_"
TIME_PREFIX     = "TIME_EORTC_QLQ_C30__3_1__ITEM_"
SUBSCALE_PREFIX = "EORTC_QLQ_C30__3_1__SUBSCALE_"


def normalize_regimen(regimen, sep="/"):
    """
    @brief Normalize a chemotherapy regimen name by sorting its components alphabetically.

    @param regimen  Raw regimen string, e.g. 'Navelbine/Cisplatin'.
    @param sep     Component separator used in the regimen string (default '/').
    @return regimen string with components sorted and joined by sep.
    """
    components = [c.strip() for c in regimen.split(sep)]
    return sep.join(sorted(components))


def normalize_regimen_column(df, col=TREATMENT_COL, sep="/"):
    """
    @brief Apply regimen normalization to an entire DataFrame column in-place.

    @param df   DataFrame containing the regimen column.
    @param col  Name of the column holding raw regimen strings.
    @param sep  Component separator (default '/').
    @return The same DataFrame with the regimen column overwritten by normalized values.
    """
    if col not in df.columns:
        return df
    original_unique = df[col].nunique()
    df[col] = df[col].fillna("").apply(
        lambda r: normalize_regimen(r, sep) if r else r
    )
    normalized_unique = df[col].nunique()
    print(f"  regimen unique values: {original_unique} -> {normalized_unique} after normalization")
    return df


def classify_columns(df):
    """
    @brief Group columns into semantic categories.

    @param df  Source DataFrame.
    @return Dict mapping category name -> list of column names.
    """
    items = [c for c in df.columns
             if c.startswith(ITEM_PREFIX) and not c.startswith("TIME_")]
    item_times = [c for c in df.columns if c.startswith(TIME_PREFIX)]
    subscales  = [c for c in df.columns if c.startswith(SUBSCALE_PREFIX)]
    meta = [c for c in [ID_COL, GENDER_COL, DIAGNOSIS_COL, TREATMENT_COL,
                        "Erstes_Treffen", DECEASED_COL, DEATH_DATE_COL,
                        EXAM_DATE_COL]
            if c in df.columns]
    return {
        "meta":       meta,
        "items":      items,
        "item_times": item_times,
        "subscales":  subscales,
    }


def _is_deceased_series(s):
    """
    @brief Robustly coerce a 'deceased' column into a boolean Series.

    @param s  Raw column (numeric, string, or boolean encoded).
    @return Boolean Series of equal length.
    """
    return s.astype(str).str.strip().str.lower().isin(
        ["1", "1.0", "true", "ja", "yes", "y"]
    )


def summarise(df, name):
    """
    @brief Print high-level dataset statistics.

    @param df    DataFrame to summarise.
    @param name  Label used in the printed header.
    @return None
    """
    cols = classify_columns(df)
    n_rows = len(df)
    n_patients = df[ID_COL].nunique() if ID_COL in df.columns else None

    print("=" * 72)
    print(f"  {name}")
    print("=" * 72)
    print(f"rows                : {n_rows}")
    if n_patients is not None:
        print(f"unique patients     : {n_patients}")
        if n_patients:
            print(f"rows / patient (mean): {n_rows / n_patients:.1f}")
    print()
    print(f"EORTC item columns  : {len(cols['items'])} value + "
          f"{len(cols['item_times'])} time")
    print(f"EORTC subscales     : {len(cols['subscales'])}")
    print(f"meta columns        : {len(cols['meta'])}")
    print()

    if TREATMENT_COL in df.columns:
        print("Chemotherapy regimens (per visit):")
        regimen_counts = df[TREATMENT_COL].value_counts(dropna=False)
        regimen_nr = 1
        for regimen, count in regimen_counts.items():
            print(f"  {regimen_nr}. {str(regimen):<35} {count:>5}")
            regimen_nr += 1
        print()

    if GENDER_COL in df.columns:
        print("Gender (per patient):")
        for k, v in df.drop_duplicates(ID_COL)[GENDER_COL].value_counts(dropna=False).items():
            print(f"  {str(k):<10} {v}")
        print()

    if DIAGNOSIS_COL in df.columns:
        print("Top-10 initial diagnoses (per patient):")
        for k, v in (df.drop_duplicates(ID_COL)[DIAGNOSIS_COL]
                       .value_counts(dropna=False).head(10).items()):
            print(f"  {str(k)[:50]:<50} {v}")
        print()

    if DECEASED_COL in df.columns and n_patients:
        pat = df.drop_duplicates(ID_COL)
        dead = _is_deceased_series(pat[DECEASED_COL])
        if dead.sum() == 0 and DEATH_DATE_COL in pat.columns:
            dead = pat[DEATH_DATE_COL].notna()
        n_dead = int(dead.sum())
        print(f"deceased            : {n_dead} / {n_patients} "
              f"({100 * n_dead / n_patients:.1f}%)")
        print()

    if cols["subscales"]:
        miss = df[cols["subscales"]].isna().mean().sort_values(ascending=False)
        print("Top-5 subscale missing rates:")
        for k, v in miss.head(5).items():
            short = k.replace(SUBSCALE_PREFIX, "")
            print(f"  {short:<32} {100 * v:5.1f}%")
        print()


def plot_regimen_distribution(df, out_dir):
    """
    @brief Bar chart of visits per chemotherapy regimen.

    @param df       DataFrame with treatment column.
    @param out_dir  Directory where the figure is written.
    @return Path to the saved figure or None if column missing.
    """
    if TREATMENT_COL not in df.columns:
        return None
    series = df[TREATMENT_COL]
    series = series.replace("", pd.NA).fillna("NaN")
    counts = series.value_counts(dropna=False)
    if "NaN" in counts.index:
        nan_count = counts.pop("NaN")
        counts["NaN"] = nan_count

    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_ylabel("Visits")
    ax.set_xlabel("regimen")
    ax.set_title("Visits per chemotherapy regimen")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = out_dir / "regimen_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_gender_and_diagnosis(df, out_dir):
    """
    @brief Side-by-side bar charts of gender and top diagnoses.

    @param df       DataFrame with gender / diagnosis columns.
    @param out_dir  Directory where the figure is written.
    @return Path to the saved figure or None if columns missing.
    """
    if GENDER_COL not in df.columns and DIAGNOSIS_COL not in df.columns:
        return None
    pat = df.drop_duplicates(ID_COL)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if GENDER_COL in pat.columns:
        pat[GENDER_COL].value_counts(dropna=False).plot(
            kind="bar", ax=axes[0], color="#55A868"
        )
        axes[0].set_title("Gender")
        axes[0].set_ylabel("Patients")
        axes[0].tick_params(axis="x", rotation=0)

    if DIAGNOSIS_COL in pat.columns:
        top = pat[DIAGNOSIS_COL].value_counts(dropna=False).head(10)
        top.plot(kind="barh", ax=axes[1], color="#8172B2")
        axes[1].invert_yaxis()
        axes[1].set_title("Top-10 initial diagnoses")
        axes[1].set_xlabel("Patients")

    plt.tight_layout()
    path = out_dir / "gender_and_diagnosis.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_visits_per_patient(df, out_dir):
    """
    @brief Histogram of number of rows (visits) per patient.

    @param df       DataFrame with patient ID column.
    @param out_dir  Directory where the figure is written.
    @return Path to the saved figure.
    """
    visits = df.groupby(ID_COL).size()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(visits.values, bins=30, color="#55A868", edgecolor="black")
    ax.set_xlabel("Visits per patient")
    ax.set_ylabel("Patients")
    ax.set_title(
        f"Visit cadence  (median={int(visits.median())}, "
        f"max={int(visits.max())}, mean={visits.mean():.1f})"
    )
    plt.tight_layout()

    path = out_dir / "visits_per_patient.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_subscale_distributions(df, out_dir):
    """
    @brief Boxplot of EORTC subscale values across all observations.

    @param df       DataFrame with subscale columns.
    @param out_dir  Directory where the figure is written.
    @return Path to the saved figure or None if no subscales found.
    """
    cols = classify_columns(df)["subscales"]
    if not cols:
        return None
    short = [c.replace(SUBSCALE_PREFIX, "") for c in cols]
    data = [df[c].dropna().values for c in cols]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(data, labels=short, showfliers=False)
    ax.set_ylabel("Score (0-100)")
    ax.set_title("EORTC QLQ-C30 subscale distributions")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = out_dir / "subscale_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run(raw_path, prep_dir, fig_dir, include_mixed):
    """
    @brief Execute the full visualisation pipeline.

    @param raw_path       Path to the raw input CSV.
    @param prep_dir       Directory holding preprocessed splits.
    @param fig_dir        Directory in which figures are written.
    @param include_mixed  Whether to also summarise the mixed splits.
    @return None
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    print(f"\nLoading raw data from {raw_path}")
    raw = pd.read_csv(raw_path)
    raw = normalize_regimen_column(raw)
    summarise(raw, f"RAW: {raw_path.name}")

    saved.append(plot_regimen_distribution(raw,    fig_dir))
    saved.append(plot_gender_and_diagnosis(raw,   fig_dir))
    saved.append(plot_visits_per_patient(raw,     fig_dir))
    saved.append(plot_subscale_distributions(raw, fig_dir))

    train_p = prep_dir / "train.csv"
    test_p  = prep_dir / "test.csv"
    if train_p.exists():
        df = normalize_regimen_column(pd.read_csv(train_p))
        summarise(df, f"PREPROCESSED TRAIN: {train_p.name}")
    if test_p.exists():
        df = normalize_regimen_column(pd.read_csv(test_p))
        summarise(df, f"PREPROCESSED TEST:  {test_p.name}")

    if include_mixed:
        mtrain = prep_dir / "train_mixed.csv"
        mtest  = prep_dir / "test_mixed.csv"
        if mtrain.exists():
            df = normalize_regimen_column(pd.read_csv(mtrain))
            summarise(df, f"MIXED TRAIN: {mtrain.name}")
        if mtest.exists():
            df = normalize_regimen_column(pd.read_csv(mtest))
            summarise(df, f"MIXED TEST:  {mtest.name}")

    print("Saved figures:")
    for p in saved:
        if p is not None:
            print(f"  {p}")


def main():
    """
    @brief Parse CLI arguments and run the visualisation.

    @return None
    """
    parser = argparse.ArgumentParser(
        description="Exploratory plots and stats for the chemotherapy QoL dataset."
    )
    parser.add_argument("--raw",      type=Path, default=RAW_PATH,
                        help="Path to raw CSV.")
    parser.add_argument("--prep-dir", type=Path, default=PREP_DIR,
                        help="Preprocessed data directory.")
    parser.add_argument("--fig-dir",  type=Path, default=FIG_DIR,
                        help="Output directory for figures.")
    parser.add_argument("--no-mixed", action="store_true",
                        help="Skip the mixed-split summaries.")
    args = parser.parse_args()
    run(args.raw, args.prep_dir, args.fig_dir,
        include_mixed=not args.no_mixed)


if __name__ == "__main__":
    main()