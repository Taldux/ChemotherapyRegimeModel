"""
@file evaluate_generators.py
@brief Compute realism and structural metrics for one or two semi-synthetic
       test_mixed CSVs, and optionally generate diagnostic plots. Usage:

    uv run python evaluate_generators.py --files test_mixed_handcrafted.csv
    uv run python evaluate_generators.py --files test_mixed_handcrafted.csv test_mixed_crn.csv
    uv run python evaluate_generators.py --files test_mixed_handcrafted.csv --plots
"""

import argparse
import itertools
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


TARGET_COL    = "time_to_last_days"
TREATMENT_COL = "Chemotherapieregime_enc"
Y_COL_PREFIX  = "Y"
TREATMENTS    = [0, 1, 2]
CSV_DIR       = "src/semi-eval"
PLOT_DIR      = "src/semi-eval/plots"


def _label_from_path(path: str) -> str:
    """
    @brief Derive a display label from a CSV filename.

    Strips the 'test_mixed_' prefix and the '.csv' suffix.

    @param path  Path to the CSV file.
    @return Label string, e.g. 'handcrafted' for 'test_mixed_handcrafted.csv'.
    """
    name = os.path.basename(path)
    match = re.match(r"test_mixed_(.+)\.csv$", name)
    return match.group(1) if match else os.path.splitext(name)[0]


def _factual_predictions(df: pd.DataFrame) -> np.ndarray:
    """
    @brief Extract the synthetic outcome under each visit's actually assigned treatment.

    @param df  DataFrame with TREATMENT_COL and Y0...Y{K-1} columns.
    @return Array of shape (n_rows,) with the per-row factual synthetic outcome.
    """
    t = df[TREATMENT_COL].astype(int).values
    return np.array([df[f"{Y_COL_PREFIX}{t[i]}"].iloc[i] for i in range(len(df))])


def compute_factual_realism(df: pd.DataFrame) -> dict:
    """
    @brief Compute how closely the synthetic factual outcome matches the real outcome.

    @param df  Mixed dataset DataFrame.
    @return Dict with rmse, mae, bias, pearson_r, spearman_r metrics.
    """
    y_real = df[TARGET_COL].values.astype(float)
    y_fact = _factual_predictions(df).astype(float)
    err    = y_fact - y_real

    return {
        "factual_rmse": float(np.sqrt(np.mean(err ** 2))),
        "factual_mae":  float(np.mean(np.abs(err))),
        "factual_bias": float(np.mean(err)),
        "pearson_r":    float(stats.pearsonr(y_fact, y_real).statistic),
        "spearman_r":   float(stats.spearmanr(y_fact, y_real).statistic),
    }


def compute_distributional_realism(df: pd.DataFrame) -> dict:
    """
    @brief Compare the marginal distributions of factual and synthetic outcomes.

    @param df  Mixed dataset DataFrame.
    @return Dict with ks_statistic, ks_pvalue, real_mean, syn_mean, real_std, syn_std.
    """
    y_real = df[TARGET_COL].values.astype(float)
    y_fact = _factual_predictions(df).astype(float)
    ks     = stats.ks_2samp(y_fact, y_real)

    return {
        "ks_statistic": float(ks.statistic),
        "ks_pvalue":    float(ks.pvalue),
        "real_mean":    float(y_real.mean()),
        "syn_mean":     float(y_fact.mean()),
        "real_std":     float(y_real.std()),
        "syn_std":      float(y_fact.std()),
    }


def compute_counterfactual_structure(df: pd.DataFrame) -> dict:
    """
    @brief Summarise the structure of the synthetic potential outcomes Y_k.

    @param df  Mixed dataset DataFrame.
    @return Dict with per-arm means, pairwise ATEs, pairwise ITE stds, best-arm distribution.
    """
    out = {}

    for k in TREATMENTS:
        out[f"mean_Y{k}"] = float(df[f"{Y_COL_PREFIX}{k}"].mean())
        out[f"std_Y{k}"]  = float(df[f"{Y_COL_PREFIX}{k}"].std())

    for a, b in itertools.combinations(TREATMENTS, 2):
        tau = (df[f"{Y_COL_PREFIX}{b}"] - df[f"{Y_COL_PREFIX}{a}"]).values
        out[f"ATE_T{a}_to_T{b}"]     = float(tau.mean())
        out[f"ITE_std_T{a}_to_T{b}"] = float(tau.std())

    y_matrix = df[[f"{Y_COL_PREFIX}{k}" for k in TREATMENTS]].values
    best_arm = np.argmax(y_matrix, axis=1)
    for k in TREATMENTS:
        out[f"best_arm_share_T{k}"] = float(np.mean(best_arm == k))

    return out


def compare_generators(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    """
    @brief Compare two generators on the same underlying patients.

    Both DataFrames must be aligned by row (same Patienten_ID + visit_nr order).

    @param df_a  First generator's mixed dataset.
    @param df_b  Second generator's mixed dataset.
    @return Dict with per-arm outcome correlations and best-arm agreement metrics.
    """
    out = {}

    for k in TREATMENTS:
        ya = df_a[f"{Y_COL_PREFIX}{k}"].values.astype(float)
        yb = df_b[f"{Y_COL_PREFIX}{k}"].values.astype(float)
        out[f"corr_Y{k}"] = float(stats.pearsonr(ya, yb).statistic)

    best_a = np.argmax(df_a[[f"{Y_COL_PREFIX}{k}" for k in TREATMENTS]].values, axis=1)
    best_b = np.argmax(df_b[[f"{Y_COL_PREFIX}{k}" for k in TREATMENTS]].values, axis=1)
    out["best_arm_agreement"] = float(np.mean(best_a == best_b))

    rank_agreements = []
    for i in range(len(df_a)):
        ra = stats.rankdata([df_a.iloc[i][f"{Y_COL_PREFIX}{k}"] for k in TREATMENTS])
        rb = stats.rankdata([df_b.iloc[i][f"{Y_COL_PREFIX}{k}"] for k in TREATMENTS])
        if np.std(ra) > 0 and np.std(rb) > 0:
            rank_agreements.append(stats.kendalltau(ra, rb).statistic)
    out["mean_kendall_tau"] = float(np.nanmean(rank_agreements)) if rank_agreements else float("nan")

    return out


def print_metrics(name: str, metrics: dict) -> None:
    """
    @brief Print a labelled block of metrics to stdout.

    @param name     Section name shown in the header.
    @param metrics  Dict of metric_name -> float.
    """
    print("\n" + "=" * 60)
    print(f"  {name}")
    print("=" * 60)
    width = max(len(k) for k in metrics)
    for key, value in metrics.items():
        print(f"  {key:<{width}}  {value:>12.4f}")


def _save_fig(fig, filename: str) -> None:
    """
    @brief Save a matplotlib figure to PLOT_DIR and close it.

    @param fig       Matplotlib Figure object.
    @param filename  Filename (without directory) under PLOT_DIR.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_factual_scatter(df: pd.DataFrame, label: str) -> None:
    """
    @brief Scatter the synthetic factual outcome against the real outcome.

    A perfect generator places all points on the diagonal y=x.

    @param df     Mixed dataset DataFrame.
    @param label  Display label used in title and filename.
    """
    y_real = df[TARGET_COL].values.astype(float)
    y_fact = _factual_predictions(df).astype(float)
    lim    = max(y_real.max(), y_fact.max())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_real, y_fact, alpha=0.4, s=18)
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
    ax.set_xlabel("Real time_to_last_days")
    ax.set_ylabel("Synthetic Y under factual treatment")
    ax.set_title(f"Factual realism — {label}")
    ax.legend()
    _save_fig(fig, f"factual_scatter_{label}.png")


def plot_outcome_distributions(df: pd.DataFrame, label: str) -> None:
    """
    @brief Compare the marginal distributions of factual synthetic and real outcomes.

    @param df     Mixed dataset DataFrame.
    @param label  Display label used in title and filename.
    """
    y_real = df[TARGET_COL].values.astype(float)
    y_fact = _factual_predictions(df).astype(float)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(y_real.max(), y_fact.max()), 40)
    ax.hist(y_real, bins=bins, alpha=0.5, label="Real", color="tab:blue")
    ax.hist(y_fact, bins=bins, alpha=0.5, label="Synthetic (factual)", color="tab:orange")
    ax.set_xlabel("time_to_last_days")
    ax.set_ylabel("Count")
    ax.set_title(f"Outcome distribution — {label}")
    ax.legend()
    _save_fig(fig, f"outcome_distribution_{label}.png")


def plot_per_arm_distributions(df: pd.DataFrame, label: str) -> None:
    """
    @brief Compare distributions of the three synthetic potential outcomes Y_k.

    @param df     Mixed dataset DataFrame.
    @param label  Display label used in title and filename.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    max_val = max(df[f"{Y_COL_PREFIX}{k}"].max() for k in TREATMENTS)
    bins    = np.linspace(0, max_val, 40)
    for k in TREATMENTS:
        ax.hist(df[f"{Y_COL_PREFIX}{k}"].values, bins=bins, alpha=0.45,
                label=f"Y{k}")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Count")
    ax.set_title(f"Per-arm synthetic outcomes — {label}")
    ax.legend()
    _save_fig(fig, f"per_arm_distribution_{label}.png")


def plot_best_arm_share(df: pd.DataFrame, label: str) -> None:
    """
    @brief Bar plot of the share of visits where each arm is the true best.

    @param df     Mixed dataset DataFrame.
    @param label  Display label used in title and filename.
    """
    y_matrix = df[[f"{Y_COL_PREFIX}{k}" for k in TREATMENTS]].values
    best_arm = np.argmax(y_matrix, axis=1)
    shares   = [float(np.mean(best_arm == k)) for k in TREATMENTS]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"T{k}" for k in TREATMENTS], shares, color="tab:green")
    for k, share in enumerate(shares):
        ax.text(k, share + 0.01, f"{share:.1%}", ha="center")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of visits where arm is true best")
    ax.set_title(f"Best-arm distribution — {label}")
    _save_fig(fig, f"best_arm_share_{label}.png")


def plot_comparison_scatter(df_a: pd.DataFrame, df_b: pd.DataFrame,
                             label_a: str, label_b: str) -> None:
    """
    @brief Scatter Y_k of generator 1 against Y_k of generator 2 for each arm.

    @param df_a    First generator's mixed dataset.
    @param df_b    Second generator's mixed dataset.
    @param label_a Display label for the first generator.
    @param label_b Display label for the second generator.
    """
    fig, axes = plt.subplots(1, len(TREATMENTS), figsize=(5 * len(TREATMENTS), 5))

    for ax, k in zip(axes, TREATMENTS):
        ya = df_a[f"{Y_COL_PREFIX}{k}"].values.astype(float)
        yb = df_b[f"{Y_COL_PREFIX}{k}"].values.astype(float)
        lim = max(ya.max(), yb.max())
        ax.scatter(ya, yb, alpha=0.4, s=18)
        ax.plot([0, lim], [0, lim], "r--", lw=1)
        ax.set_xlabel(f"{label_a} — Y{k}")
        ax.set_ylabel(f"{label_b} — Y{k}")
        ax.set_title(f"Arm T{k}")
    fig.suptitle(f"Per-arm outcome agreement: {label_a} vs {label_b}")
    fig.tight_layout()
    _save_fig(fig, f"comparison_per_arm_{label_a}_vs_{label_b}.png")


def plot_metrics_table(columns: list, label: str = "metrics") -> None:
    """
    @brief Render a metrics table as an image.

    Each column is a (label, metrics_dict) tuple. Metric keys are unioned
    across columns so that all rows align. Missing values render as '-'.

    @param columns  List of (column_label, metrics_dict) tuples.
    @param label    Suffix for the output filename.
    """
    all_keys = []
    for _, metrics in columns:
        for key in metrics:
            if key not in all_keys:
                all_keys.append(key)

    cell_text = []
    for key in all_keys:
        row = [key]
        for _, metrics in columns:
            if key in metrics:
                row.append(f"{metrics[key]:.4f}")
            else:
                row.append("-")
        cell_text.append(row)

    col_labels = ["Metric"] + [label for label, _ in columns]

    fig_height = max(3, 0.35 * len(all_keys) + 1)
    fig_width  = 4 + 2.5 * len(columns)
    fig, ax    = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)

    for j in range(len(col_labels)):
        cell = table[(0, j)]
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#dddddd")

    fig.suptitle(f"Metrics summary — {label}", y=0.98)
    _save_fig(fig, f"metrics_table_{label}.png")


def generate_single_plots(df: pd.DataFrame, label: str) -> None:
    """
    @brief Produce all single-generator plots for one mixed dataset.

    @param df     Mixed dataset DataFrame.
    @param label  Display label used in titles and filenames.
    """
    print(f"\n[{label}] Generating plots…")
    plot_factual_scatter(df, label)
    plot_outcome_distributions(df, label)
    plot_per_arm_distributions(df, label)
    plot_best_arm_share(df, label)


def generate_comparison_plots(df_a: pd.DataFrame, df_b: pd.DataFrame,
                               label_a: str, label_b: str) -> None:
    """
    @brief Produce all cross-generator comparison plots.

    @param df_a    First generator's mixed dataset.
    @param df_b    Second generator's mixed dataset.
    @param label_a Display label for the first generator.
    @param label_b Display label for the second generator.
    """
    print("\n[comparison] Generating plots…")
    plot_comparison_scatter(df_a, df_b, label_a, label_b)


def collect_metrics(df: pd.DataFrame) -> dict:
    """
    @brief Collect all single-generator metrics into one dict.

    @param df  Mixed dataset DataFrame.
    @return Dict combining factual, distributional, and structural metrics.
    """
    out = {}
    out.update(compute_factual_realism(df))
    out.update(compute_distributional_realism(df))
    out.update(compute_counterfactual_structure(df))
    return out


def evaluate_single(path: str, make_plots: bool) -> tuple:
    """
    @brief Load one mixed CSV, print metrics, and optionally generate plots.

    @param path        Path to the test_mixed_*.csv file.
    @param make_plots  If True, save diagnostic plots under PLOT_DIR.
    @return Tuple (label, DataFrame, metrics_dict).
    """
    if not os.path.exists(path):
        print(f"  ERROR: file not found: {path}")
        sys.exit(1)

    label = _label_from_path(path)
    df    = pd.read_csv(path)
    print(f"\n[{label}] Loaded {path}: {len(df)} rows, "
          f"{df['Patienten_ID'].nunique()} patients")

    print_metrics(f"{label} | Factual realism",          compute_factual_realism(df))
    print_metrics(f"{label} | Distributional realism",   compute_distributional_realism(df))
    print_metrics(f"{label} | Counterfactual structure", compute_counterfactual_structure(df))

    metrics = collect_metrics(df)
    if make_plots:
        generate_single_plots(df, label)

    return label, df, metrics


def main():
    """
    @brief CLI entry point: evaluate one or two semi-synthetic generators.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate semi-synthetic test_mixed CSVs against the factual outcomes."
    )
    parser.add_argument(
        "--files", nargs="+", required=True,
        help="One or two test_mixed_*.csv filenames (relative paths resolve to CSV_DIR)."
    )
    parser.add_argument(
        "--plots", action="store_true",
        help="Generate diagnostic plots and metrics tables under PLOT_DIR."
    )
    args = parser.parse_args()

    if len(args.files) not in (1, 2):
        print("  ERROR: provide one or two --files paths.")
        sys.exit(1)

    args.files = [os.path.join(CSV_DIR, f) if not os.path.isabs(f) else f
                  for f in args.files]

    results = [evaluate_single(p, args.plots) for p in args.files]
    labels  = [r[0] for r in results]
    dfs     = [r[1] for r in results]
    metrics = [r[2] for r in results]

    if len(dfs) == 2:
        if len(dfs[0]) != len(dfs[1]):
            print("\n  WARNING: generators have different row counts "
                  f"({len(dfs[0])} vs {len(dfs[1])}); skipping comparison.")
        else:
            comparison = compare_generators(dfs[0], dfs[1])
            print_metrics(f"{labels[0]} vs {labels[1]}", comparison)
            if args.plots:
                generate_comparison_plots(dfs[0], dfs[1], labels[0], labels[1])
                plot_metrics_table(
                    list(zip(labels, metrics)) + [(f"{labels[0]} vs {labels[1]}", comparison)],
                    label=f"{labels[0]}_vs_{labels[1]}",
                )

    if args.plots and len(dfs) == 1:
        plot_metrics_table(list(zip(labels, metrics)), label=labels[0])


if __name__ == "__main__":
    main()