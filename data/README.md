# data/

All data files for the project. Raw data is never modified; preprocessing scripts write to sub-directories.

## Structure

```
data/
  raw/                        Original source data (read-only)
    c30_test_mypatientreport.csv   EORTC QLQ-C30 patient records
  preprocessed/               Output of src/preprocess/preprocess.py
  synthetic_preprocessed/     Output of generate_synthetic.py
  dropped_regimes.txt         Treatments excluded for having <40 samples
```

## `preprocessed/`

Output from `src/preprocess/preprocess.py` and `semisynth_gen.py` on the real patient cohort.

- `train.csv` / `test.csv`: full dataset (real + counterfactual rows); roughly 1800 / 600 rows, 21 columns
- `train_mixed.csv` / `test_mixed.csv`: real rows only, with all potential outcomes Y₀, Y₁, Y₂
- `meta.json`: feature list, treatment mapping, split info
- `encoding_reference.txt`: human-readable label encoding
- `scaler.pkl` / `label_encoders.pkl` / `target_scaler.pkl`: fitted sklearn transformers
- `eval_results.csv`: latest evaluation metrics from `src/train.py`

## `synthetic_preprocessed/`

Output from `generate_synthetic.py` using fully synthetic patients (no real data required).

- `train.csv` / `test.csv`: synthetic feature + treatment + outcome rows
- `train_mixed.csv` / `test_mixed.csv`: synthetic real rows with all potential outcomes
- `meta.json`: metadata for the synthetic cohort
- `eval_results.csv`: evaluation metrics on synthetic data

## Features (19 columns)

16 EORTC QLQ-C30 subscales (physical, role, emotional, cognitive, social functioning; global QoL; fatigue; nausea; pain; dyspnea; sleep; appetite; constipation; diarrhea; financial; taste) + `prev_treatment` + `time_in_treatment_days` + `time_in_treatment_months`.
