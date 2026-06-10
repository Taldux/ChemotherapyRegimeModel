# src/

Main source package. All modules are designed to be run from the repository root via `uv run python`.

## Files

- `train.py`: training pipeline: loads data, fits all models, evaluates on real rows, and generates plots
- `plots.py`: plotting functions for PEHE, ATE error, ITE distributions, CI coverage, and policy comparison
- `visualise.py`: exploratory analysis: treatment distributions, deceased patients, synthetic outcome shapes

## Sub-packages

- `models/`: causal model implementations (S-Learner, T-Learner, BOWL, Causal Forest, DynamicDML)
- `preprocess/`: raw data preprocessing and semi-synthetic dataset generation
- `crn/`: Counterfactual Recurrent Network (GRU-based sequence model)
- `semi-eval/`: semi-synthetic data evaluation