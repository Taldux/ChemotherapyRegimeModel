# src/crn/

Counterfactual Recurrent Network (CRN): a GRU-based sequence model for treatment effect estimation over time.

## Files

- `crn.py`: model definition (`CRN`, `GradReversal`, `CRNDataset`) and full train/eval pipeline
- `preprocess.py`: CRN-specific preprocessing: sequence building, normalisation, encoder fitting
- `plots.py`: scatter, residual, boxplot, trajectory, and counterfactual distribution figures

## Running

```bash
# Preprocess (writes to src/crn/preprocessed/)
uv run python -m src.crn.preprocess

# Train
uv run python -m src.crn.crn

# Plot results
uv run python -m src.crn.plots
```

## Artifacts (`preprocessed/`)

- `crn_weights.pt`: trained CRN model weights
- `t_learner_arm*.pt`: per-arm T-Learner weights (baseline comparison)
- `train.csv` / `test.csv`: preprocessed sequence data
- `meta.json`: feature names, treatment mapping, scaler parameters
