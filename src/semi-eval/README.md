# evaluate_generators.py
Stand-alone script that evaluates one or two semi-synthetic `test_mixed_*.csv`
files against the factual outcomes. Run from the repository root via
`uv run python`.

## Usage
- One generator: `uv run python evaluate_generators.py --files test_mixed_handcrafted.csv`
- Two generators: `uv run python evaluate_generators.py --files test_mixed_handcrafted.csv test_mixed_crn.csv`
- Add `--plots` to save diagnostic plots and a metrics table under `src/semi-eval/plots/`.

CSV files are resolved relative to `src/semi-eval/`. The suffix after
`test_mixed_` is used as the display label in plot titles and filenames.

## Plots (with `--plots`)
- `factual_scatter_<label>.png`, `outcome_distribution_<label>.png`,
  `per_arm_distribution_<label>.png`, `best_arm_share_<label>.png`
- `comparison_per_arm_<a>_vs_<b>.png` when two files are provided
- `metrics_table_<label>.png` summarising all numeric metrics