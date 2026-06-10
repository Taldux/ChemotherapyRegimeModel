"""
@file preprocess.py
@brief Preprocessing pipeline for the CRN module.

Loads raw EORTC QLQ-C30 visit data, normalises regime names, filters to the
top-N chemotherapy regimes, computes the regression target, encodes categoricals,
imputes missing values, splits by patient, fits scalers, and saves all artefacts
to src/crn/preprocessed/.

Run from the repo root:
    python -m src.crn.preprocess
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler


DATA_PATH   = "data/raw/c30_test_mypatientreport.csv"
OUTPUT_DIR  = "src/crn/preprocessed"
RANDOM_SEED = 42
TEST_SIZE   = 0.2

SUBSCALE_COLS = [
    "EORTC_QLQ_C30__3_1__SUBSCALE_Physical_Functioning",
    "EORTC_QLQ_C30__3_1__SUBSCALE_Role_Functioning",
    "EORTC_QLQ_C30__3_1__SUBSCALE_Emotional_Functioning",
    "EORTC_QLQ_C30__3_1__SUBSCALE_Cognitive_Functioning",
    "EORTC_QLQ_C30__3_1__SUBSCALE_Global_Quality_of_Life",
    "EORTC_QLQ_C30__3_1__SUBSCALE_Fatigue",
    "EORTC_QLQ_C30__3_1__SUBSCALE_Nausea___Vomiting",
    "EORTC_QLQ_C30__3_1__SUBSCALE_Pain",
    "EORTC_QLQ_C30__3_1__SUBSCALE_Sleep_Disturbances",
    "EORTC_QLQ_C30__3_1__SUBSCALE_Appetite_Loss",
]

TARGET_COL    = "time_to_last_days"
TREATMENT_COL = "Chemotherapieregime_enc"
TOP_N_REGIMES = 3


def normalize_regime(regime: str, sep: str = "/") -> str:
    """
    @brief Sort the components of a regime name alphabetically so
           order-variant spellings map to one canonical form.

    @param regime  Raw regime string, e.g. 'Navelbine/Cisplatin'.
    @param sep     Component separator. Default '/'.
    @return Alphabetically sorted, sep-joined regime string.
    """
    components = [c.strip() for c in regime.split(sep)]
    return sep.join(sorted(components))


def normalize_regime_column(
    df: pd.DataFrame,
    col: str = "Chemotherapieregime",
    sep: str = "/",
) -> pd.DataFrame:
    """
    @brief Apply regime normalisation row-wise to a DataFrame column in-place.

    @param df   DataFrame containing the regime column.
    @param col  Name of the column holding raw regime strings.
    @param sep  Component separator. Default '/'.
    @return Same DataFrame with the regime column overwritten by normalised values.
    """
    original_unique = df[col].nunique()
    df[col] = df[col].fillna("").apply(lambda r: normalize_regime(r, sep) if r else r)
    normalized_unique = df[col].nunique()
    print(f"  Regime unique values: {original_unique} → {normalized_unique} after normalisation")
    return df


def load_data(path: str) -> pd.DataFrame:
    """
    @brief Load raw CSV data and print a brief summary.

    @param path  Filesystem path to the CSV file.
    @return Raw DataFrame as read from the CSV.
    """
    df = pd.read_csv(path, low_memory=False)
    print(f"  Shape: {df.shape}  |  Patienten: {df['Patienten_ID'].nunique()}")
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Convert date columns to datetime and warn about unparseable entries.

    @param df  DataFrame with raw string date columns 'examdate',
               'Erstes_Treffen', and 'Sterbedatum'.
    @return DataFrame with those columns converted to datetime64 (NaT on failure).
    """
    for col in ["examdate", "Erstes_Treffen", "Sterbedatum"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            n_nat = int(df[col].isna().sum())
            if n_nat:
                print(f"  Warning: {col} has {n_nat} NaT values")
    return df


def filter_top_n_regimes(df: pd.DataFrame, n: int = TOP_N_REGIMES) -> tuple:
    """
    @brief Retain only rows whose regime is among the n most frequent regimes.

    @param df  DataFrame with a normalised 'Chemotherapieregime' column.
    @param n   Number of top regimes to keep.
    @return Tuple (filtered_df, list_of_top_regime_names).
    """
    counts   = df["Chemotherapieregime"].value_counts()
    top_n    = counts.head(n).index.tolist()
    n_before = len(df)
    df       = df[df["Chemotherapieregime"].isin(top_n)].copy()
    print(f"  Top-{n} regimes: {top_n}")
    print(f"  Rows: {n_before} → {len(df)}  |  Patients: {df['Patienten_ID'].nunique()}")
    return df, top_n


def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Compute the regression target 'time_to_last_days' for every row.

    The target is the number of days between the current visit and the
    patient's final recorded visit (last examdate = 0, earlier visits > 0).

    @param df  DataFrame with 'Patienten_ID' and 'examdate' columns.
    @return DataFrame with an additional integer column 'time_to_last_days'.
    """
    last_exam         = df.groupby("Patienten_ID")["examdate"].transform("max")
    df[TARGET_COL]    = (last_exam - df["examdate"]).dt.days.astype(int)
    assert (df[TARGET_COL] >= 0).all(), "Negative target: check date ordering!"
    print(f"  {TARGET_COL}: min={df[TARGET_COL].min()}  "
          f"max={df[TARGET_COL].max()}  mean={df[TARGET_COL].mean():.1f}")
    return df


def remove_single_visit_patients(df: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Remove all rows belonging to patients with only one recorded visit.

    Patients with a single visit have a target of zero and provide no sequential
    signal for the recurrent model.

    @param df  DataFrame with 'Patienten_ID' and 'examdate' columns.
    @return DataFrame with single-visit patients removed entirely.
    """
    visit_counts = df.groupby("Patienten_ID")["examdate"].transform("nunique")
    n_before     = df["Patienten_ID"].nunique()
    df           = df[visit_counts > 1].copy()
    removed      = n_before - df["Patienten_ID"].nunique()
    print(f"  Patients: {n_before} → {df['Patienten_ID'].nunique()}  "
          f"({removed} single-visit removed)")
    return df


def sort_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Sort by patient and visit date, then add a zero-based visit index.

    @param df  DataFrame with 'Patienten_ID' and 'examdate' columns.
    @return DataFrame sorted by ['Patienten_ID', 'examdate'] with a new
            integer column 'visit_nr' (0-based per patient).
    """
    df             = df.sort_values(["Patienten_ID", "examdate"]).reset_index(drop=True)
    df["visit_nr"] = df.groupby("Patienten_ID").cumcount().astype(int)
    max_visits     = int(df.groupby("Patienten_ID")["visit_nr"].max().max()) + 1
    avg_visits     = df.groupby("Patienten_ID")["visit_nr"].max().mean()
    print(f"  Max visits: {max_visits}  |  Avg visits: {avg_visits:.1f}")
    return df


def encode_categoricals(
    df: pd.DataFrame,
    doc_path: str = f"{OUTPUT_DIR}/encoding_reference.txt",
) -> tuple:
    """
    @brief Label-encode gender, diagnosis, and chemo regime columns.

    Writes a human-readable encoding reference to doc_path so that integer
    codes can be traced back to their original string values.

    @param df        DataFrame with 'Geschlecht', 'Initiale_Diagnose', and
                     'Chemotherapieregime' columns.
    @param doc_path  Path for the encoding reference text file. Parent directory
                     is created if it does not exist.
    @return Tuple (df_with_enc_columns, dict_of_label_encoders).
    """
    columns = [
        ("Geschlecht",          "Geschlecht_enc"),
        ("Initiale_Diagnose",   "Initiale_Diagnose_enc"),
        ("Chemotherapieregime", TREATMENT_COL),
    ]

    encoders = {}
    for col, enc_col in columns:
        le             = LabelEncoder()
        df[enc_col]    = le.fit_transform(df[col].fillna("unbekannt")).astype(int)
        encoders[col]  = le
        mapping        = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
        print(f"  {col}: {mapping}")

    lines = ["CATEGORICAL ENCODING REFERENCE",
             "=" * 60,
             "Method : sklearn LabelEncoder (alphabetical order)",
             "Note   : missing values were filled with 'unbekannt' before encoding",
             ""]

    for col, enc_col in columns:
        le = encoders[col]
        lines += [
            f"Column : {col}  →  {enc_col}",
            "-" * 40,
        ]
        for cls, idx in zip(le.classes_, le.transform(le.classes_)):
            lines.append(f"  {int(idx):>3}  =  {cls}")
        lines.append("")

    os.makedirs(os.path.dirname(doc_path), exist_ok=True)
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Encoding reference written to: {doc_path}")

    return df, encoders


def impute_missing(df: pd.DataFrame, subscale_cols: list) -> pd.DataFrame:
    """
    @brief Impute missing subscale values with forward-fill then global median fallback.

    @param df            DataFrame containing the subscale columns.
    @param subscale_cols Names of the subscale columns to impute.
    @return DataFrame with no missing values in the specified columns.
    """
    n_before          = int(df[subscale_cols].isna().sum().sum())
    df[subscale_cols] = (
        df.groupby("Patienten_ID")[subscale_cols]
        .transform(lambda x: x.ffill())
    )
    medians           = df[subscale_cols].median()
    df[subscale_cols] = df[subscale_cols].fillna(medians)
    n_after           = int(df[subscale_cols].isna().sum().sum())
    print(f"  NaN: {n_before} → {n_after}")

    for col in subscale_cols:
        out_of_range = int(((df[col] < 0) | (df[col] > 100)).sum())
        if out_of_range:
            print(f"  Warning: {col} has {out_of_range} values outside [0, 100]")
    return df


def select_columns(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    @brief Reduce the DataFrame to only the columns required for model training.

    Always keeps 'Patienten_ID', 'examdate', the target column, and the treatment
    column. Deduplicates the final column list while preserving order.

    @param df           Full DataFrame after all preprocessing steps.
    @param feature_cols Additional feature columns to retain.
    @return Slimmed DataFrame containing only the relevant columns.
    """
    keep = ["Patienten_ID", "examdate", TARGET_COL, TREATMENT_COL] + feature_cols
    seen, keep_unique = set(), []
    for c in keep:
        if c not in seen and c in df.columns:
            keep_unique.append(c)
            seen.add(c)
    return df[keep_unique].copy()


def patient_split(df: pd.DataFrame) -> tuple:
    """
    @brief Split the dataset into train and test sets at the patient level.

    @param df  DataFrame with a 'Patienten_ID' column.
    @return Tuple (df_train, df_test) with no patient overlap.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(df, groups=df["Patienten_ID"].values))
    df_train = df.iloc[train_idx].copy()
    df_test  = df.iloc[test_idx].copy()

    overlap = set(df_train["Patienten_ID"]) & set(df_test["Patienten_ID"])
    assert len(overlap) == 0, f"Overlap: {len(overlap)} patients in both train and test!"

    print(f"  Train: {len(df_train):>6} rows  |  {df_train['Patienten_ID'].nunique()} patients")
    print(f"  Test:  {len(df_test):>6} rows  |  {df_test['Patienten_ID'].nunique()} patients")
    return df_train, df_test


def fit_scalers(df_train: pd.DataFrame, subscale_cols: list) -> tuple:
    """
    @brief Fit a StandardScaler for features and a separate one for the target.

    @param df_train      Training split DataFrame.
    @param subscale_cols Names of the subscale feature columns.
    @return Tuple (feature_scaler, target_scaler) fitted on training data only.
    """
    scaler        = StandardScaler()
    target_scaler = StandardScaler()
    scaler.fit(df_train[subscale_cols])
    target_scaler.fit(df_train[[TARGET_COL]])
    print(f"  Scalers fitted on {len(df_train)} training rows")
    example_col = subscale_cols[0].split("_")[-1]
    print(f"  Example {example_col}: mean={scaler.mean_[0]:.2f}, std={scaler.scale_[0]:.2f}")
    return scaler, target_scaler


def save_artifacts(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    scaler: StandardScaler,
    target_scaler: StandardScaler,
    encoders: dict,
    top_n_regimes: list,
    feature_cols: list,
    output_dir: str,
) -> None:
    """
    @brief Persist all preprocessing artefacts to disk.

    Writes train.csv, test.csv, pickled scalers and label encoders, and a
    JSON metadata file describing the feature configuration and dataset statistics.

    @param df_train       Training split DataFrame.
    @param df_test        Test split DataFrame.
    @param scaler         Fitted feature scaler.
    @param target_scaler  Fitted target scaler.
    @param encoders       Fitted LabelEncoder instances keyed by column name.
    @param top_n_regimes  The retained top-N regime names.
    @param feature_cols   Ordered list of model input feature column names.
    @param output_dir     Directory path where all files are written.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_train.to_csv(f"{output_dir}/train.csv", index=False)
    df_test.to_csv(f"{output_dir}/test.csv",   index=False)

    with open(f"{output_dir}/scaler.pkl",        "wb") as f: pickle.dump(scaler,        f)
    with open(f"{output_dir}/target_scaler.pkl", "wb") as f: pickle.dump(target_scaler, f)
    with open(f"{output_dir}/label_encoders.pkl","wb") as f: pickle.dump(encoders,      f)

    max_seq = int(df_train["visit_nr"].max() + 1)
    meta = {
        "feature_cols":   feature_cols,
        "subscale_cols":  SUBSCALE_COLS,
        "target_col":     TARGET_COL,
        "treatment_col":  TREATMENT_COL,
        "top_n_regimes":  top_n_regimes,
        "n_features":     len(feature_cols),
        "max_seq_len":    max_seq,
        "train_patients": int(df_train["Patienten_ID"].nunique()),
        "test_patients":  int(df_test["Patienten_ID"].nunique()),
    }
    with open(f"{output_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main_preprocess() -> None:
    """
    @brief Execute the full CRN preprocessing pipeline end-to-end.

    Steps:
      1.  Load raw CSV data.
      2.  Parse date columns.
      3.  Normalise chemotherapy regime names (order-invariant).
      4.  Filter to Top-N regimes.
      5.  Compute per-row regression target.
      6.  Remove single-visit patients.
      7.  Sort and index visit sequences.
      8.  Label-encode categorical columns.
      9.  Impute missing subscale values.
      10. Select relevant columns.
      11. Split into patient-level train / test sets.
      12. Fit feature and target scalers on training data.
      13. Save all artefacts to OUTPUT_DIR.
    """
    df = load_data(DATA_PATH)
    df = parse_dates(df)
    df = normalize_regime_column(df)
    df, top_n_regimes = filter_top_n_regimes(df)
    df = compute_target(df)
    df = remove_single_visit_patients(df)
    df = sort_sequences(df)

    df, encoders = encode_categoricals(df)

    feature_cols = SUBSCALE_COLS + ["Geschlecht_enc", "Initiale_Diagnose_enc", "visit_nr"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    df = impute_missing(df, SUBSCALE_COLS)
    df = select_columns(df, feature_cols)
    print(f"  Final columns: {list(df.columns)}")

    df_train, df_test = patient_split(df)

    for frame in (df_train, df_test):
        frame["visit_nr"] = frame["visit_nr"].astype(int)

    scaler, target_scaler = fit_scalers(df_train, SUBSCALE_COLS)

    save_artifacts(
        df_train, df_test, scaler, target_scaler,
        encoders, top_n_regimes, feature_cols, OUTPUT_DIR,
    )


if __name__ == "__main__":
    main_preprocess()
