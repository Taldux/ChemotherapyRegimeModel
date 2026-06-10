# tests/

pytest unit tests for the core source modules. All tests use synthetic in-memory data, no data files required.

## Running

```bash
uv run pytest tests/
uv run pytest tests/ -v
uv run pytest tests/test_metrics.py 
```

## Files

- `conftest.py`: shared fixtures: `rng` (seeded RNG), `three_arm_data` (60-sample dataset), `linear_base` (sklearn LinearRegression)
- `test_metrics.py` (16 tests): covers `calculate_pehe`, `calculate_ate_error`, `evaluate_semi_synthetic`, and `evaluate_clinical_threshold_policy`
- `test_tlearner.py` (10 tests): covers `TLearner.fit`, per-arm model creation, ITE direction, antisymmetry, and unseen-treatment errors
- `test_slearner.py` (9 tests): covers `SLearner.fit`, treatment concatenation, ITE direction, antisymmetry, and self-ITE = 0
- `test_panel_utils.py` (9 tests): covers `balance_panel`: LOCF padding, truncation, X=None, W provided, median default, min-2 floor
- `test_preprocess.py` (13 tests): covers `normalize_regime` (sorting, whitespace, idempotency) and `normalize_regime_column`
