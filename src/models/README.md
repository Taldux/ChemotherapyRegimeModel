# src/models/

Causal inference model implementations. All models share a common interface: `fit(X, t, y)`, `predict_outcome(X, treatment)`, `predict(X, from_t, to_t)`.

## Files

- `slearner.py`: S-Learner: single `M(X, T) → Y` model where treatment is concatenated as a feature
- `tlearner.py`: T-Learner: one `M_k(X) → Y` model per treatment arm
- `causal_forest.py`: Causal Forest: doubly-robust CATE via `econml.CausalForestDML`; supports confidence intervals
- `dynamic_dml.py`: DynamicDML: panel-data CATE estimator supporting cross-sectional and longitudinal data
- `bowl.py`: BOWL: outcome-weighted pairwise classification for policy learning using plurality voting
- `lstm_model.py`: LSTM sequence model for time-series treatment effect estimation
- `metrics.py`: evaluation utilities: PEHE, ATE error, `evaluate_semi_synthetic`, clinical threshold policy
- `panel_utils.py`: `balance_panel`: LOCF padding and truncation for unequal-length longitudinal panels

## Evaluation Metrics (`metrics.py`)

- **PEHE**: Precision in Estimation of Heterogeneous Effects: $\sqrt{\frac{1}{n}\sum(\hat\tau_i - \tau_i)^2}$
- **ATE error**: $|\bar{\hat\tau} - \bar\tau|$
- **Clinical threshold policy**: only switches treatment when predicted improvement exceeds a threshold
