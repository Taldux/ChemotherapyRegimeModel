# Causal Inference for Dynamic Chemotherapy Treatment Regimes

Estimates optimal chemotherapy treatment regimes for cancer patients using causal inference. Five methods are trained and compared on semi synthetic data generated from real EORTC QLQ-C30 quality of life measurements.

## Methods

- **S-Learner**: single model with treatment concatenated as a feature
- **T-Learner**: separate model per treatment arm
- **BOWL**: outcome-weighted policy learning via pairwise classification
- **Causal Forest**: doubly-robust CATE estimator via `econml`
- **DynamicDML**: panel-data CATE estimator for longitudinal data
- **CRN**: Counterfactual Recurrent Network (GRU-based sequence model)

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)
- Doxygen
- Usable EORTC QLQ-C30 dataset
  
uv dependencies:
- `numpy`, `pandas`, `scikit-learn`, `scipy`
- `econml` — Causal Forest and DynamicDML estimators
- `torch` — LSTM and CRN models
- `matplotlib`, `seaborn` — plotting
- `pytest` — running tests (optional, dev only)

All dependencies are declared in `pyproject.toml` and installed via `uv sync`.

## Quick Start

```bash
# Install dependencies
uv sync
```

## Running

Preprocessed data and plots are not stored in the repository. Run the following steps in order to reconstruct everything from the raw data (`data/raw/c30_test_mypatientreport.csv` (also not included in the repository)).

**Step 1: Preprocess real patient data**

```bash
uv run python -m src.preprocess.preprocess
# Outputs: data/preprocessed/train.csv, test.csv, train_mixed.csv, test_mixed.csv, meta.json, encoding_reference.txt, *.pkl
```

**Step 2: Preprocess data for the CRN model**

```bash
uv run python -m src.crn.preprocess
# Outputs: src/crn/preprocessed/
```

**Step 3: Generate semi-synthetic counterfactuals**

```bash
uv run python -m src.preprocess.semisynth_gen
# Outputs: data/preprocessed/train_mixed.csv, test_mixed.csv
```

**Step 4: Generate fully synthetic data** (optional, for `--data synth`)

```bash
uv run python -m src.preprocess.synth_gen --n_train 12000 --n_test 3000 --seed 42
# Outputs: data/synthetic_preprocessed/
```

**Step 5: Train and evaluate all models**

```bash
# Semi-synthetic data (default)
uv run python main.py --data semi

# Fully synthetic data
uv run python main.py --data synth
```

To train only specific models:
```bash
uv run python main.py --data semi --models s_learner causal_forest
```

## Running Tests

```bash
uv run pytest tests/
```

## Generating Documentation

```bash
# Requires Doxygen installed (C:\Program Files\doxygen\bin)
doxygen Doxyfile
# Output: docs/html/index.html
```

## Repository Layout

```
data/
  raw/          Raw EORTC QLQ-C30 patient data (source of truth)
  preprocessed/ Generated — run src.preprocess.preprocess + semisynth_gen
  synthetic_preprocessed/ Generated — run src.preprocess.synth_gen
src/
  models/       Causal model implementations
  preprocess/   Data preprocessing and semi-synthetic generation
  crn/          Counterfactual Recurrent Network module
  train.py      Training pipeline and evaluation
  plots.py      Visualisation utilities
  visualise.py  Data analysis helpers
tests/          pytest unit tests
docs/           Generated Doxygen HTML documentation
plots/          Generated — produced by main.py
```
