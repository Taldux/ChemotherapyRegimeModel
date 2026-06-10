# src/preprocess/

Data preprocessing and dataset generation pipeline. Run from the repository root.

## Files

- `preprocess.py`: reads the raw CSV, normalises regime names, encodes categoricals, imputes missing values, and splits train/test at patient level
- `semisynth_gen.py`: takes real data and generates counterfactual potential outcomes Y₀, Y₁, Y₂ for each patient visit using a hidden state model
- `synth_gen.py`: generates fully synthetic patients with no real data required; used by `generate_synthetic.py`

## Output Files

- `train.csv` / `test.csv`: full feature + treatment + outcome rows (real + counterfactual)
- `train_mixed.csv` / `test_mixed.csv`: real rows only with all K potential outcomes
- `meta.json`: column names, treatment mapping, scaler parameters
- `encoding_reference.txt`: human-readable label encoding reference
- `scaler.pkl` / `label_encoders.pkl`: fitted sklearn transformers
