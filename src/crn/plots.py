"""
@file plots.py
@brief Compare factual outcomes (time_to_last_days) vs. CRN counterfactual predictions.

Generates six diagnostic figures for training and test data:
  1. Scatter: factual vs. CRN prediction per treatment arm
  2. Residual histograms (prediction - factual) per treatment arm
  3. Boxplot of absolute errors per treatment arm (factual rows only)
  4. Patient trajectories: factual vs. all counterfactuals over visit number
  5. Counterfactual outcome distributions
  6. Summary table with MAE, RMSE, R² per arm and split

Run from the repo root:
    python -m src.crn.plots

Inputs  : src/crn/preprocessed/train_mixed.csv, test_mixed.csv
Outputs : src/crn/plots/*.png
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


INPUT_DIR       = "src/crn/preprocessed"
PLOTS_DIR       = "src/crn/plots"
TARGET_COL      = "time_to_last_days"
TREATMENT_COL   = "Chemotherapieregime_enc"
CF_COLS         = ["Y0", "Y1", "Y2"]
TREATMENT_NAMES = {0: "Regime 0", 1: "Regime 1", 2: "Regime 2"}

COLORS = {
    0: "#2563eb",   # blue
    1: "#dc2626",   # red
    2: "#16a34a",   # green
}
BG_COLOR    = "#fafafa"
GRID_COLOR  = "#e5e7eb"
TEXT_COLOR  = "#1f2937"
ACCENT_GRAY = "#6b7280"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   BG_COLOR,
    "axes.edgecolor":   GRID_COLOR,
    "axes.grid":        True,
    "grid.color":       GRID_COLOR,
    "grid.alpha":       0.6,
    "text.color":       TEXT_COLOR,
    "axes.labelcolor":  TEXT_COLOR,
    "xtick.color":      ACCENT_GRAY,
    "ytick.color":      ACCENT_GRAY,
    "font.size":        10,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "figure.titlesize": 16,
})


def load_data(input_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    @brief Load train_mixed.csv and test_mixed.csv from the preprocessed directory.

    @param input_dir  Directory containing train_mixed.csv and test_mixed.csv.
    @return Tuple (df_train, df_test).
    """
    train = pd.read_csv(os.path.join(input_dir, "train_mixed.csv"))
    test  = pd.read_csv(os.path.join(input_dir, "test_mixed.csv"))
    return train, test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    @brief Compute regression metrics between ground-truth and predicted values.

    @param y_true  Array of observed values.
    @param y_pred  Array of predicted values.
    @return Dict with keys 'MAE', 'RMSE', 'R²', and 'Bias'.
    """
    residuals = y_pred - y_true
    mae  = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R²": r2, "Bias": np.mean(residuals)}


def plot_scatter(
    df: pd.DataFrame,
    split_name: str,
    ax: plt.Axes,
    treatment: int,
) -> None:
    """
    @brief Draw a factual-vs-prediction scatter plot for one treatment arm.

    All rows are shown as faint grey points; rows whose factual treatment matches
    the arm are highlighted in the arm's colour. Includes a diagonal reference
    line and a metric annotation.

    @param df          Panel DataFrame with TARGET_COL, TREATMENT_COL, and Y{t} columns.
    @param split_name  Label for the split ('Train' or 'Test') used in text annotations.
    @param ax          Matplotlib Axes to draw onto.
    @param treatment   Treatment arm index (0, 1, or 2).
    """
    col    = f"Y{treatment}"
    y_true = df[TARGET_COL].values
    y_pred = df[col].values

    ax.scatter(y_true, y_pred, s=8, alpha=0.15, color=ACCENT_GRAY, zorder=1)

    mask = df[TREATMENT_COL] == treatment
    ax.scatter(
        y_true[mask], y_pred[mask],
        s=18, alpha=0.5, color=COLORS[treatment],
        edgecolors="white", linewidths=0.3, zorder=2,
    )

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", alpha=0.4, linewidth=0.8)

    m = compute_metrics(y_true[mask], y_pred[mask])
    ax.text(
        0.05, 0.95,
        f"Factual:\nMAE={m['MAE']:.1f}  RMSE={m['RMSE']:.1f}\nR²={m['R²']:.3f}",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Factual (days)")
    ax.set_ylabel(f"{col} Prediction (days)")
    ax.set_title(f"{TREATMENT_NAMES[treatment]}")


def fig_scatter(df_train: pd.DataFrame, df_test: pd.DataFrame) -> plt.Figure:
    """
    @brief Create a 2×3 grid of factual-vs-prediction scatter plots.

    Rows correspond to train/test splits; columns correspond to treatment arms.

    @param df_train  Training DataFrame with counterfactual columns.
    @param df_test   Test DataFrame with counterfactual columns.
    @return Matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Factual vs. CRN Prediction", fontweight="bold", y=0.98)

    for i, t in enumerate(range(3)):
        plot_scatter(df_train, "Train", axes[0, i], t)
        plot_scatter(df_test,  "Test",  axes[1, i], t)

    axes[0, 0].set_ylabel("Train\nPrediction (days)")
    axes[1, 0].set_ylabel("Test\nPrediction (days)")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def fig_residuals(df_train: pd.DataFrame, df_test: pd.DataFrame) -> plt.Figure:
    """
    @brief Create residual histograms (prediction - factual) for each arm and split.

    @param df_train  Training DataFrame with counterfactual columns.
    @param df_test   Test DataFrame with counterfactual columns.
    @return Matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Residuals (CRN Prediction − Factual)", fontweight="bold", y=0.98)

    for row, (df, label) in enumerate([(df_train, "Train"), (df_test, "Test")]):
        for col_idx, t in enumerate(range(3)):
            ax        = axes[row, col_idx]
            mask      = df[TREATMENT_COL] == t
            residuals = df.loc[mask, f"Y{t}"].values - df.loc[mask, TARGET_COL].values

            ax.hist(residuals, bins=40, color=COLORS[t], alpha=0.7,
                    edgecolor="white", linewidth=0.5)
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.axvline(np.mean(residuals), color=COLORS[t], linestyle="-",
                       linewidth=1.5, alpha=0.9)

            ax.set_title(f"{label} – {TREATMENT_NAMES[t]}")
            ax.set_xlabel("Residual (days)")
            ax.set_ylabel("Count")

            ax.text(
                0.95, 0.95,
                f"μ={np.mean(residuals):.1f}\nσ={np.std(residuals):.1f}",
                transform=ax.transAxes, fontsize=9, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def fig_boxplot(df_train: pd.DataFrame, df_test: pd.DataFrame) -> plt.Figure:
    """
    @brief Boxplot of absolute errors per treatment arm (factual rows only).

    @param df_train  Training DataFrame with counterfactual columns.
    @param df_test   Test DataFrame with counterfactual columns.
    @return Matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Absolute Error |CRN − Factual| (factual treatment only)",
                 fontweight="bold", y=1.0)

    for ax, (df, label) in zip(axes, [(df_train, "Train"), (df_test, "Test")]):
        data, labels, colors = [], [], []
        for t in range(3):
            mask    = df[TREATMENT_COL] == t
            abs_err = np.abs(df.loc[mask, f"Y{t}"].values - df.loc[mask, TARGET_COL].values)
            data.append(abs_err)
            labels.append(TREATMENT_NAMES[t])
            colors.append(COLORS[t])

        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False,
                        medianprops=dict(color="white", linewidth=1.5))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)

        ax.set_ylabel("Absolute Error (days)")
        ax.set_title(label)

    fig.tight_layout()
    return fig


def fig_trajectories(
    df: pd.DataFrame,
    split_name: str,
    n_patients: int = 6,
) -> plt.Figure:
    """
    @brief Plot factual and all CRN counterfactual outcomes over visit number per patient.

    Selects the n_patients patients with the most visits (at least 3).
    Each subplot shows factual trajectory (black) and Y0/Y1/Y2 (coloured dashes).
    Factual treatment at each visit is marked with a colour-coded scatter point.

    @param df          Panel DataFrame with visit_nr, TARGET_COL, TREATMENT_COL, Y0-Y2.
    @param split_name  Label for the split used in the figure title.
    @param n_patients  Maximum number of patient subplots to show.
    @return Matplotlib Figure.
    """
    pids = df["Patienten_ID"].unique()
    candidates = [
        (pid, len(df[df["Patienten_ID"] == pid]))
        for pid in pids
        if len(df[df["Patienten_ID"] == pid]) >= 3
    ]
    candidates.sort(key=lambda x: -x[1])
    selected = [c[0] for c in candidates[:n_patients]]

    n_cols = min(3, len(selected))
    n_rows = (len(selected) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle(
        f"Patient Trajectories ({split_name}): Factual vs. Counterfactuals",
        fontweight="bold", y=1.0,
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, pid in enumerate(selected):
        r, c = divmod(idx, n_cols)
        ax   = axes[r, c]
        grp  = df[df["Patienten_ID"] == pid].sort_values("visit_nr")
        visits = grp["visit_nr"].values

        ax.plot(visits, grp[TARGET_COL].values, "ko-", linewidth=2,
                markersize=5, label="Factual", zorder=3)

        for t in range(3):
            ax.plot(visits, grp[f"Y{t}"].values, color=COLORS[t],
                    linewidth=1.2, alpha=0.7, linestyle="--", marker="s",
                    markersize=3, label=f"Y{t} ({TREATMENT_NAMES[t]})")

        for _, row in grp.iterrows():
            t_actual = int(row[TREATMENT_COL])
            ax.scatter(row["visit_nr"], row[TARGET_COL],
                       color=COLORS[t_actual], s=40, zorder=4,
                       edgecolors="black", linewidths=0.5)

        ax.set_xlabel("Visit")
        ax.set_ylabel("Days")
        ax.set_title(f"Patient {pid}", fontsize=10)
        if idx == 0:
            ax.legend(fontsize=7, loc="best")

    for idx in range(len(selected), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].set_visible(False)

    fig.tight_layout()
    return fig


def fig_summary_table(df_train: pd.DataFrame, df_test: pd.DataFrame) -> plt.Figure:
    """
    @brief Render a tabular summary of MAE, RMSE, R², and Bias per arm and split.

    Metrics are computed using factual rows only for each treatment arm.

    @param df_train  Training DataFrame with counterfactual columns.
    @param df_test   Test DataFrame with counterfactual columns.
    @return Matplotlib Figure containing the formatted table.
    """
    rows = []
    for split_name, df in [("Train", df_train), ("Test", df_test)]:
        for t in range(3):
            mask = df[TREATMENT_COL] == t
            n    = mask.sum()
            if n == 0:
                continue
            m = compute_metrics(
                df.loc[mask, TARGET_COL].values,
                df.loc[mask, f"Y{t}"].values,
            )
            rows.append({
                "Split":     split_name,
                "Treatment": TREATMENT_NAMES[t],
                "N":         n,
                "MAE":       f"{m['MAE']:.2f}",
                "RMSE":      f"{m['RMSE']:.2f}",
                "R²":        f"{m['R²']:.4f}",
                "Bias":      f"{m['Bias']:+.2f}",
            })

    tbl = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 2 + 0.4 * len(rows)))
    ax.axis("off")
    fig.suptitle("Summary: CRN vs. Factual (factual treatment only)",
                 fontweight="bold", y=0.95)

    table = ax.table(
        cellText=tbl.values,
        colLabels=tbl.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    for j in range(len(tbl.columns)):
        table[0, j].set_facecolor("#374151")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(rows) + 1):
        color = "#f3f4f6" if i % 2 == 0 else "white"
        for j in range(len(tbl.columns)):
            table[i, j].set_facecolor(color)

    fig.tight_layout()
    return fig


def fig_cf_distributions(df_train: pd.DataFrame, df_test: pd.DataFrame) -> plt.Figure:
    """
    @brief Plot overlapping histograms of Y0, Y1, Y2, and factual outcome distributions.

    @param df_train  Training DataFrame with counterfactual columns.
    @param df_test   Test DataFrame with counterfactual columns.
    @return Matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Distribution of Counterfactual Outcomes Y0, Y1, Y2",
                 fontweight="bold", y=1.0)

    for ax, (df, label) in zip(axes, [(df_train, "Train"), (df_test, "Test")]):
        for t in range(3):
            vals = df[f"Y{t}"].values
            ax.hist(vals, bins=40, color=COLORS[t], alpha=0.45,
                    label=f"Y{t} (μ={vals.mean():.1f})", edgecolor="white", linewidth=0.4)

        fv = df[TARGET_COL].values
        ax.hist(fv, bins=40, color="black", alpha=0.25, histtype="step",
                linewidth=1.5, label=f"Factual (μ={fv.mean():.1f})")

        ax.set_xlabel("Days")
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend(fontsize=9)

    fig.tight_layout()
    return fig


def main() -> None:
    """
    @brief Run all CRN diagnostic plots and save them to PLOTS_DIR.
    """
    df_train, df_test = load_data(INPUT_DIR)
    print(f"  Train: {len(df_train)} rows, Test: {len(df_test)} rows")

    for col in CF_COLS:
        if col not in df_train.columns or col not in df_test.columns:
            print(f"ERROR: Column '{col}' not found. Run crn.py first.")
            sys.exit(1)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    figures = {
        "01_scatter":          fig_scatter(df_train, df_test),
        "02_residuals":        fig_residuals(df_train, df_test),
        "03_boxplot":          fig_boxplot(df_train, df_test),
        "04_traj_test":        fig_trajectories(df_test, "Test", n_patients=6),
        "05_traj_train":       fig_trajectories(df_train, "Train", n_patients=6),
        "06_cf_distributions": fig_cf_distributions(df_train, df_test),
        "07_summary":          fig_summary_table(df_train, df_test),
    }

    for name, fig in figures.items():
        path = os.path.join(PLOTS_DIR, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  > {path}")

    print(f"\nPlots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
