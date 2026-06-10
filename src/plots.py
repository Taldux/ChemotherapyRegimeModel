import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import itertools


# ── Model registry ─────────────────────────────────────────────────────────────
# Add a new model here and every plot function picks it up automatically.
# Key   : the model_key used as dict key in data passed to plot functions.
# Value : (display_name, hex_color, optional_linestyle)
MODEL_STYLES: dict = {
    "slearner":       ("S-Learner",       "#2ecc71", "-"),
    "tlearner":       ("T-Learner",       "#e74c3c", "-"),
    "bowl":           ("BOWL",            "#9b59b6", "-"),
    "causal_forest":  ("Causal Forest",   "#3498db", "-"),
    "dynamic_dml":    ("DynamicDML",      "#e67e22", "-"),
    "lstm_s":         ("LSTM S-Learner",  "#008b8b", "--"),
    "lstm_t":         ("LSTM T-Learner",  "#8b008b", "--"),
}

# ── Treatment palette ──────────────────────────────────────────────────────────
# Colors and names are derived per-index.  Extend TREATMENT_COLORS to support
# more than the default number of treatments.
TREATMENT_COLORS: list = [
    "#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED",
    "#0891B2", "#BE185D", "#065F46",
]


def _treatment_color(idx: int) -> str:
    """
    @brief Return a hex color for a treatment index, cycling the palette.

    @param idx  Zero-based treatment index.
    @return Hex color string.
    """
    return TREATMENT_COLORS[idx % len(TREATMENT_COLORS)]


def _treatment_pairs(n_treatments: int, sep: str = " vs ") -> list:
    """
    @brief Generate all unique treatment pair labels for n_treatments treatments.

    @param n_treatments  Total number of treatments.
    @param sep           Separator between pair components (default ' vs ').
    @return List of labels such as ['T0 vs T1', 'T0 vs T2', 'T1 vs T2'].
    """
    return [f"T{a}{sep}T{b}" for a, b in itertools.combinations(range(n_treatments), 2)]


def _active_models(data_dict: dict) -> list:
    """
    @brief Return MODEL_STYLES keys present in data_dict, in registry order.

    @param data_dict  Dict whose keys are model keys.
    @return Ordered list of model keys found in data_dict.
    """
    return [k for k in MODEL_STYLES if k in data_dict]


def _deduplicate_treatment_pairs(labels: list,
                                  separators: tuple = (" vs ", " -> ", " → ", "->")) -> list:
    """
    @brief Drop b→a if a→b is already present in the label list.

    @param labels      Raw list of treatment-pair labels.
    @param separators  Substrings used as pair separators.
    @return Deduplicated labels in original order.
    """
    seen, unique = set(), []
    for label in labels:
        matched = False
        for sep in separators:
            if sep in label:
                a, b = label.split(sep, 1)
                pair = frozenset([a.strip(), b.strip()])
                if pair not in seen:
                    seen.add(pair)
                    unique.append(label)
                matched = True
                break
        if not matched and label not in seen:
            seen.add(label)
            unique.append(label)
    return unique


def save_plot(filename: str) -> None:
    """
    @brief Save the current matplotlib figure to the plots/ directory and close it.

    @param filename  Output filename including extension, e.g. 'pehe.png'.
    """
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, bbox_inches="tight", dpi=300)
    print(f"Plot saved to: {path}")
    plt.close()


def plot_comparative_ite_distribution(tau_true: np.ndarray, models: dict,
                                       title: str, filename: str) -> None:
    """
    @brief KDE plot comparing the ITE distribution of ground truth versus all models.

    @param tau_true  Ground-truth individual treatment effects.
    @param models    Dict {model_key: tau_array}; None values are skipped.
    @param title     Figure title.
    @param filename  Output filename.
    """
    plt.figure(figsize=(12, 6))
    sns.kdeplot(tau_true, color="black", label="Ground Truth",
                fill=True, alpha=0.2, linewidth=3)

    for key, tau in models.items():
        if tau is None:
            continue
        name, color, ls = MODEL_STYLES.get(key, (key, "gray", "--"))
        sns.kdeplot(tau, color=color, label=name, linewidth=2, linestyle=ls)

    plt.axvline(0, color="gray", linestyle="--")
    plt.xlim(-75, 75)
    plt.title(title)
    plt.xlabel("Treatment Effect (Outcome Change)")
    plt.ylabel("Density")
    plt.legend()
    save_plot(filename)


def plot_comparative_scatter(tau_true: np.ndarray, models: dict,
                              title: str, filename: str) -> None:
    """
    @brief Scatter plots of predicted vs true ITE, one panel per model.

    @param tau_true  Ground-truth individual treatment effects.
    @param models    Dict {model_key: tau_array}.
    @param title     Super-title for the figure.
    @param filename  Output filename.
    """
    active = [(k, v) for k, v in models.items() if v is not None]
    if not active:
        print("plot_comparative_scatter: no model data provided, skipping.")
        return

    n_plots = len(active)
    n_cols = min(n_plots, 3)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 7 * n_rows), sharey=True)
    axes = np.array(axes).flatten()
    for ax in axes[n_plots:]:
        ax.set_visible(False)

    for ax, (key, tau_pred) in zip(axes, active):
        name, color, _ = MODEL_STYLES.get(key, (key, "gray", "-"))
        ax.scatter(tau_true, tau_pred, alpha=0.5, color=color, label=name)
        ax.plot([tau_true.min(), tau_true.max()],
                [tau_true.min(), tau_true.max()], "k--")
        ax.set_title(f"{name} Accuracy")
        ax.set_xlabel("True Effect")
        if ax is axes[0]:
            ax.set_ylabel("Predicted Effect")

    plt.suptitle(title)
    save_plot(filename)


def plot_policy_comparison(current: float, oracle: float, models: dict,
                            filename: str, bowl_value: float | None = None) -> None:
    """
    @brief Horizontal bar chart comparing clinical gain across policies and models.

    @param current     Outcome under the observed (current) policy.
    @param oracle      Outcome under the oracle optimal policy.
    @param models      Dict {model_key: policy_outcome}; None values are skipped.
    @param filename    Output filename.
    @param bowl_value  Optional standalone BOWL policy value.
    """
    labels = ["Current"]
    values = [current]
    colors = ["#95a5a6"]

    for key, val in models.items():
        if val is None:
            continue
        name, color, _ = MODEL_STYLES.get(key, (key, "gray", "-"))
        labels.append(name)
        values.append(val)
        colors.append(color)

    if bowl_value is not None:
        labels.append("BOWL")
        values.append(bowl_value)
        colors.append("#9b59b6")

    labels.append("Oracle")
    values.append(oracle)
    colors.append("#27ae60")

    plt.figure(figsize=(max(10, 2 * len(labels)), 6))
    sns.barplot(x=labels, y=values, hue=labels, palette=colors, legend=False)
    plt.ylim(min(values) - 5, max(values) + 5)
    plt.title("Comparison of Total Clinical Gain")
    plt.ylabel("Avg Patient Outcome")
    save_plot(filename)


def plot_pehe_comparison(pehe_data: dict, treatment_labels: list,
                          filename: str = "pehe_comparison.png") -> None:
    """
    @brief Grouped bar chart of PEHE per treatment pair, one bar group per model.

    @param pehe_data         Dict {pair_label: {model_key: pehe_value}}.
    @param treatment_labels  Treatment pair labels to include.
    @param filename          Output filename.
    """
    treatment_labels = _deduplicate_treatment_pairs(treatment_labels)
    first_key  = treatment_labels[0]
    model_keys = _active_models(pehe_data[first_key])

    labels   = treatment_labels + ["AVERAGE"]
    n_models = len(model_keys)

    model_values = {}
    for mk in model_keys:
        vals = [pehe_data[lbl][mk] for lbl in treatment_labels]
        vals.append(np.mean(vals))
        model_values[mk] = vals

    x           = np.arange(len(labels))
    total_width = 0.8
    bar_width   = total_width / n_models

    fig, ax = plt.subplots(figsize=(max(14, 10 + 2 * n_models), 7))

    for i, mk in enumerate(model_keys):
        name, color, _ = MODEL_STYLES.get(mk, (mk, f"C{i}", "-"))
        offset = -total_width / 2 + bar_width * (i + 0.5)
        bars   = ax.bar(x + offset, model_values[mk], bar_width,
                        label=name, color=color, alpha=0.8)
        bars[-1].set_edgecolor("black")
        bars[-1].set_linewidth(2)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("PEHE (Lower is Better)")
    ax.set_title("PEHE Comparison Across Models")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    plt.tight_layout()
    save_plot(filename)


def plot_ate_error_comparison(ate_data: dict, treatment_labels: list,
                               filename: str = "ate_error_comparison.png") -> None:
    """
    @brief Grouped bar chart of ATE error per treatment pair, one bar group per model.

    @param ate_data          Dict {pair_label: {model_key: ate_error}}.
    @param treatment_labels  Treatment pair labels to include.
    @param filename          Output filename.
    """
    treatment_labels = _deduplicate_treatment_pairs(treatment_labels)
    first_key  = treatment_labels[0]
    model_keys = _active_models(ate_data[first_key])

    labels   = treatment_labels + ["AVERAGE"]
    n_models = len(model_keys)

    model_values = {}
    for mk in model_keys:
        vals = [ate_data[lbl][mk] for lbl in treatment_labels]
        vals.append(np.mean(vals))
        model_values[mk] = vals

    x           = np.arange(len(labels))
    total_width = 0.8
    bar_width   = total_width / n_models

    fig, ax = plt.subplots(figsize=(max(14, 10 + 2 * n_models), 7))

    for i, mk in enumerate(model_keys):
        name, color, _ = MODEL_STYLES.get(mk, (mk, f"C{i}", "-"))
        offset = -total_width / 2 + bar_width * (i + 0.5)
        bars   = ax.bar(x + offset, model_values[mk], bar_width,
                        label=name, color=color, alpha=0.8)
        bars[-1].set_edgecolor("black")
        bars[-1].set_linewidth(2)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("ATE Error (Lower is Better)")
    ax.set_title("ATE Error Comparison Across Models")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    plt.tight_layout()
    save_plot(filename)


def plot_pehe_heatmap(models: dict, treatment_names: list,
                      filename: str = "pehe_heatmap.png") -> None:
    """
    @brief Side-by-side PEHE heatmaps, one panel per model.

    @param models           Dict {model_key: pehe_matrix (np.ndarray)}; None or all-NaN skipped.
    @param treatment_names  Display name for each treatment.
    @param filename         Output filename.
    """
    def _has_finite(m):
        return m is not None and np.isfinite(m).any()

    active = [(k, m) for k, m in models.items() if _has_finite(m)]
    if not active:
        print("Skipping PEHE heatmap: no finite PEHE values available.")
        return

    finite_vals = [m[np.isfinite(m)] for _, m in active]
    vmin = min(v.min() for v in finite_vals)
    vmax = max(v.max() for v in finite_vals)
    mask = np.eye(len(treatment_names), dtype=bool)

    n_plots = len(active)
    n_cols = min(n_plots, 3)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 7 * n_rows))
    axes = np.array(axes).flatten()
    for ax in axes[n_plots:]:
        ax.set_visible(False)

    for ax, (key, data) in zip(axes, active):
        name, _, _ = MODEL_STYLES.get(key, (key, "gray", "-"))
        sns.heatmap(data, annot=True, fmt=".2f", cmap="RdYlGn_r",
                    xticklabels=treatment_names, yticklabels=treatment_names,
                    ax=ax, vmin=vmin, vmax=vmax, mask=mask,
                    cbar_kws={"label": "PEHE"})
        ax.set_title(f"{name} PEHE")
        ax.set_xlabel("To Treatment")
        ax.set_ylabel("From Treatment")

    plt.suptitle("PEHE Heatmap: Treatment Effect Estimation Error (Lower is Better)",
                 fontsize=14)
    plt.tight_layout()
    save_plot(filename)


def plot_regime_comparison_ate(ate_real: dict, ate_synth: dict,
                                model_names: list, model_keys: list,
                                filename: str = "regime_comparison_ate.png") -> None:
    """
    @brief Grouped bar chart comparing ATE error of real-trained vs synth-trained models.

    @param ate_real    Dict {pair_label: {model_key: ate_error}} from real-trained models.
    @param ate_synth   Dict {pair_label: {model_key: ate_error}} from synth-trained models.
    @param model_names Display names aligned with model_keys.
    @param model_keys  Model keys aligned with model_names.
    @param filename    Output filename.
    """
    pair_labels = _deduplicate_treatment_pairs(list(ate_real.keys()))
    categories  = pair_labels + ["AVERAGE"]
    n_models    = len(model_keys)

    real_vals, synth_vals = {}, {}
    for mk in model_keys:
        r = [ate_real[l][mk] for l in pair_labels]
        s = [ate_synth[l][mk] for l in pair_labels]
        r.append(np.mean(r))
        s.append(np.mean(s))
        real_vals[mk]  = r
        synth_vals[mk] = s

    x           = np.arange(len(categories))
    total_width = 0.85
    pair_width  = total_width / n_models
    sub_width   = pair_width * 0.45

    fig, ax = plt.subplots(figsize=(max(14, 6 + 3 * n_models), 7))

    for i, (mk, name) in enumerate(zip(model_keys, model_names)):
        center  = -total_width / 2 + pair_width * (i + 0.5)
        rects_r = ax.bar(x + center - sub_width / 2, real_vals[mk], sub_width,
                         color=f"C{i}", alpha=0.55, edgecolor="white")
        rects_s = ax.bar(x + center + sub_width / 2, synth_vals[mk], sub_width,
                         color=f"C{i}", alpha=1.0, edgecolor="white", hatch="///")
        rects_r[-1].set_edgecolor("black"); rects_r[-1].set_linewidth(1.5)
        rects_s[-1].set_edgecolor("black"); rects_s[-1].set_linewidth(1.5)

    handles = [mpatches.Patch(facecolor=f"C{i}", alpha=0.75, label=name)
               for i, name in enumerate(model_names)]
    handles += [
        mpatches.Patch(facecolor="grey", alpha=0.55, label="Trained on Real"),
        mpatches.Patch(facecolor="grey", alpha=1.0,  label="Trained on Synthetic", hatch="///"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    ax.set_ylabel("ATE Error (lower is better)")
    ax.set_title("ATE Error: Real-Trained vs Synthetic-Trained Models\n"
                 "(both evaluated on semi-synthetic test set)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=35, ha="right", fontsize=9)
    ax.axhline(0, color="grey", linewidth=0.5)
    plt.tight_layout()
    save_plot(filename)


def plot_regime_comparison_accuracy(acc_real: dict, acc_synth: dict,
                                     model_names: list, model_keys: list,
                                     n_treatments: int,
                                     filename: str = "regime_comparison_accuracy.png") -> None:
    """
    @brief Side-by-side bar chart comparing recommendation accuracy for real vs synth-trained models.

    @param acc_real      Dict {model_key: accuracy_percent} for real-trained models.
    @param acc_synth     Dict {model_key: accuracy_percent} for synth-trained models.
    @param model_names   Display names aligned with model_keys.
    @param model_keys    Model keys aligned with model_names.
    @param n_treatments  Number of treatments (used to compute the random baseline).
    @param filename      Output filename.
    """
    x     = np.arange(len(model_names))
    width = 0.35

    real_accs  = [acc_real[k]  for k in model_keys]
    synth_accs = [acc_synth[k] for k in model_keys]

    fig, ax = plt.subplots(figsize=(max(10, 2 * len(model_names)), 6))

    bars_r = ax.bar(x - width / 2, real_accs,  width, label="Trained on Real Data",
                    color="#3498db", alpha=0.8, edgecolor="white")
    bars_s = ax.bar(x + width / 2, synth_accs, width, label="Trained on Synthetic Data",
                    color="#e74c3c", alpha=0.8, edgecolor="white", hatch="///")

    for bars in (bars_r, bars_s):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

    random_baseline = 100.0 / n_treatments
    ax.axhline(random_baseline, color="grey", linestyle="--", alpha=0.5,
               label=f"Random ({random_baseline:.1f}%)")

    ax.set_ylabel("Recommendation Accuracy (%)")
    ax.set_title("Treatment Recommendation Accuracy:\n"
                 "Real-Trained vs Synthetic-Trained Models\n"
                 "(both evaluated on semi-synthetic test set)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_plot(filename)