"""
@file slearner.py
@brief S-Learner causal effect estimator.
"""

import numpy as np
from sklearn.base import clone, BaseEstimator

from src.models.panel_utils import balance_panel


class SLearner(BaseEstimator):
    """
    @brief S-Learner: single model M(X, T) -> Y for CATE estimation.

    Predicts counterfactuals by varying T while holding X fixed.

    @param base_model  sklearn-compatible regression estimator.
    """
    
    def __init__(self, base_model):

        self.base_model = base_model
        self.model = None
        self.treatment_values = None
        self._bootstrap_cache = {}
    
    def fit(self, X, t, y):
        """
        @brief Fit a single regression model on (X, T) -> Y.

        @param X  Feature matrix of shape (n_samples, n_features).
        @param t  Treatment vector of shape (n_samples,).
        @param y  Outcome vector of shape (n_samples,).
        @return self
        """
        X = np.asarray(X)
        t = np.asarray(t).ravel()
        y = np.asarray(y).ravel()

        # Store unique treatment values
        self.treatment_values = np.unique(t)
        
        # Clone the base model
        self.model = clone(self.base_model)
        
        # Concatenate treatment as a feature
        # X_with_treatment = [X, t]
        t_reshaped = t.reshape(-1, 1) if len(t.shape) == 1 else t
        X_with_treatment = np.column_stack([X, t_reshaped])
        
        # Fit single model
        self.model.fit(X_with_treatment, y)

        # Save training data for bootstrap CI.
        self._ci_train_mode = "flat"
        self._ci_train_X = X.astype(np.float64, copy=True)
        self._ci_train_t = t.astype(int, copy=True)
        self._ci_train_y = y.astype(np.float64, copy=True)
        self._bootstrap_cache = {}
        
        return self
    
    def predict_outcome(self, X, treatment):
        """
        @brief Predict E[Y(treatment) | X].

        @param X          Feature matrix of shape (n_samples, n_features).
        @param treatment  Treatment value.
        @return Predicted outcomes of shape (n_samples,).
        """
        n_samples = X.shape[0]

        # Use stored evaluation sequences for sequential (LSTM) models
        if (getattr(self.base_model, '_is_sequential', False)
                and hasattr(self, '_eval_X_seq')
                and n_samples == len(self._eval_X_seq)):
            X_seq = self._eval_X_seq.copy()
            T_seq = np.full(
                (n_samples, X_seq.shape[1], 1), treatment, dtype=np.float32
            )
            return self.model.predict(
                np.concatenate([X_seq, T_seq], axis=2)
            )

        # Default: 2D single-step prediction
        t_column = np.full((n_samples, 1), treatment)
        X_with_treatment = np.column_stack([X, t_column])
        return self.model.predict(X_with_treatment)

    def set_eval_sequences(self, X_seq):
        """
        @brief Store 3-D evaluation sequences for LSTM prediction.

        @param X_seq  Array of shape (n_eval, max_periods, n_features).
        """
        self._eval_X_seq = np.asarray(X_seq, dtype=np.float32)

    def predict(self, X, from_treatment, to_treatment):
        """
        @brief Predict ITE of switching from one treatment to another.

        @param X               Feature matrix of shape (n_samples, n_features).
        @param from_treatment  Current treatment value.
        @param to_treatment    Target treatment value.
        @return ITE estimates of shape (n_samples,).
        """
        y_to = self.predict_outcome(X, to_treatment)
        y_from = self.predict_outcome(X, from_treatment)
        
        return y_to - y_from

    def predict_effect_with_ci(self, X, from_treatment, to_treatment,
                                n_mc_samples=50, alpha=0.05,
                                ci_method="bootstrap",
                                n_bootstrap=100,
                                random_state=42):
        """
        @brief Predict ITE with confidence intervals.

        Primary path: non-parametric bootstrap over training units
        (rows for flat mode, patients for panel/LSTM mode). Fallback path:
        tree-variance (RF) or MC dropout (LSTM) when bootstrap data is absent.

        @param X               Feature matrix (2-D for RF; 3-D sequences for LSTM).
        @param from_treatment  Source treatment value.
        @param to_treatment    Target treatment value.
        @param n_mc_samples    MC Dropout samples for fallback LSTM CI.
        @param alpha           Significance level for the CI (default 0.05 → 95 %).
        @param ci_method       'bootstrap' (default) or 'heuristic'.
        @param n_bootstrap     Number of bootstrap refits.
        @param random_state    Random seed for bootstrap.
        @return Dict with keys 'effect', 'ci_lower', 'ci_upper' (n_samples,).
        """
        if ci_method == "bootstrap" and hasattr(self, "_ci_train_mode"):
            cached = self._get_cached_bootstrap_result(
                X, from_treatment, to_treatment,
                alpha=alpha,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )
            if cached is not None:
                return cached

        # Fallback to the previous heuristic implementation.
        from scipy.stats import norm as _norm
        z = _norm.ppf(1.0 - alpha / 2.0)

        base = self.model

        if hasattr(base, 'estimators_'):
            # ── Random Forest: use per-tree variance ──────────────────────────
            X = np.asarray(X)
            n = X.shape[0]
            X_from = np.column_stack([X, np.full((n, 1), from_treatment)])
            X_to   = np.column_stack([X, np.full((n, 1), to_treatment)])
            tree_from = np.array([t.predict(X_from) for t in base.estimators_])
            tree_to   = np.array([t.predict(X_to)   for t in base.estimators_])
            tree_cate = tree_to - tree_from
            effect = tree_cate.mean(axis=0)
            std    = tree_cate.std(axis=0)

        elif getattr(base, '_is_sequential', False) and hasattr(base, 'predict_with_uncertainty'):
            # ── LSTM: MC Dropout ──────────────────────────────────────────────
            X_seq = np.asarray(X, dtype=np.float32)   # (n, seq_len, n_feat)
            n, seq_len = X_seq.shape[0], X_seq.shape[1]
            T_from = np.full((n, seq_len, 1), from_treatment, dtype=np.float32)
            T_to   = np.full((n, seq_len, 1), to_treatment,   dtype=np.float32)
            mu_from, sig_from = base.predict_with_uncertainty(
                np.concatenate([X_seq, T_from], axis=2), n_samples=n_mc_samples)
            mu_to,   sig_to   = base.predict_with_uncertainty(
                np.concatenate([X_seq, T_to],   axis=2), n_samples=n_mc_samples)
            effect = mu_to - mu_from
            std    = np.sqrt(sig_from ** 2 + sig_to ** 2)

        else:
            effect = self.predict(X, from_treatment, to_treatment)
            std    = np.zeros_like(effect)

        return {
            'effect':   effect,
            'ci_lower': effect - z * std,
            'ci_upper': effect + z * std,
        }

    def _get_cached_bootstrap_result(self, X, from_treatment, to_treatment,
                                     alpha, n_bootstrap, random_state):
        """Return cached bootstrap CI for one treatment pair or compute it."""
        X_arr = np.asarray(X)
        key = (
            int(X_arr.__array_interface__["data"][0]),
            X_arr.shape,
            str(X_arr.dtype),
            float(alpha),
            int(n_bootstrap),
            int(random_state),
            tuple(int(t) for t in np.asarray(self.treatment_values).tolist()),
        )
        cache = self._bootstrap_cache.get(key)
        if cache is None:
            cache = self._bootstrap_all_pairs(
                X_arr,
                alpha=alpha,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )
            if cache is None:
                return None
            self._bootstrap_cache[key] = cache

        pair_key = (int(from_treatment), int(to_treatment))
        return cache.get(pair_key)

    def _bootstrap_all_pairs(self, X, alpha, n_bootstrap, random_state):
        """Compute bootstrap CIs for all treatment pairs in one pass."""
        if n_bootstrap < 2:
            return None

        tr_vals = [int(t) for t in np.asarray(self.treatment_values).tolist()]
        tr_vals.sort()

        if getattr(self.base_model, "_is_sequential", False):
            X_eval = np.asarray(X, dtype=np.float32)
            if X_eval.ndim != 3:
                return None
            point_preds = {
                t: self._predict_seq_outcome(self.model, X_eval, t)
                for t in tr_vals
            }
        else:
            X_eval = np.asarray(X, dtype=np.float64)
            if X_eval.ndim != 2:
                return None
            point_preds = {
                t: self._predict_flat_outcome(self.model, X_eval, t)
                for t in tr_vals
            }

        rng = np.random.default_rng(random_state)
        n_eval = len(X_eval)
        boot_outcomes = {
            t: np.empty((n_bootstrap, n_eval), dtype=np.float64)
            for t in tr_vals
        }

        if self._ci_train_mode == "seq":
            X_train = self._ci_train_X_seq_with_t
            y_train = self._ci_train_y
            n_train = len(y_train)
            if n_train == 0:
                return None
            for b in range(n_bootstrap):
                idx = rng.integers(0, n_train, size=n_train)
                m = clone(self.base_model)
                m.fit(X_train[idx], y_train[idx])
                for t in tr_vals:
                    boot_outcomes[t][b] = self._predict_seq_outcome(m, X_eval, t)
        elif self._ci_train_mode == "flat":
            X_train = self._ci_train_X
            t_train = self._ci_train_t
            y_train = self._ci_train_y
            n_train = len(y_train)
            if n_train == 0:
                return None
            for b in range(n_bootstrap):
                idx = rng.integers(0, n_train, size=n_train)
                m = clone(self.base_model)
                X_boot = np.column_stack([X_train[idx], t_train[idx]])
                m.fit(X_boot, y_train[idx])
                for t in tr_vals:
                    boot_outcomes[t][b] = self._predict_flat_outcome(m, X_eval, t)
        else:
            return None

        q_lo = alpha / 2.0
        q_hi = 1.0 - q_lo
        out = {}
        for i in range(len(tr_vals)):
            for j in range(i + 1, len(tr_vals)):
                t0, t1 = tr_vals[i], tr_vals[j]
                draws = boot_outcomes[t1] - boot_outcomes[t0]
                out[(t0, t1)] = {
                    "effect": point_preds[t1] - point_preds[t0],
                    "ci_lower": np.quantile(draws, q_lo, axis=0),
                    "ci_upper": np.quantile(draws, q_hi, axis=0),
                }
        return out

    @staticmethod
    def _predict_flat_outcome(model, X_eval, treatment):
        """Predict outcomes under one treatment for flat 2-D features."""
        t_col = np.full((len(X_eval), 1), treatment, dtype=np.float64)
        return model.predict(np.column_stack([X_eval, t_col]))

    @staticmethod
    def _predict_seq_outcome(model, X_eval, treatment):
        """Predict outcomes under one treatment for 3-D sequences."""
        n, seq_len = X_eval.shape[0], X_eval.shape[1]
        t_chan = np.full((n, seq_len, 1), treatment, dtype=np.float32)
        return model.predict(np.concatenate([X_eval, t_chan], axis=2))

    def fit_panel(self, Y, T, X, groups, max_periods=None):
        """
        @brief Fit on longitudinal panel data.

        Balances unequal-length panels then delegates to fit().
        For LSTM base models, reshapes into 3-D sequences.

        @param Y            Outcomes of shape (n_total,).
        @param T            Treatments of shape (n_total,).
        @param X            Features of shape (n_total, d_x).
        @param groups       Group (patient) IDs of shape (n_total,).
        @param max_periods  Target periods per group; defaults to median.
        @return self
        """
        Y = np.asarray(Y, dtype=float).ravel()
        T = np.asarray(T).ravel()
        X = np.asarray(X, dtype=float)
        groups = np.asarray(groups)

        Y, T, X, groups, _ = balance_panel(
            Y, T, X, groups, max_periods=max_periods
        )

        # Sequential model (e.g. LSTM): reshape into patient-level sequences
        if getattr(self.base_model, '_is_sequential', False):
            unique_groups = np.unique(groups)
            n_groups = len(unique_groups)
            n_periods = len(Y) // n_groups

            X_3d = X.reshape(n_groups, n_periods, -1)
            T_3d = T.reshape(n_groups, n_periods, 1).astype(float)
            Y_2d = Y.reshape(n_groups, n_periods)

            # S-Learner: include treatment as a feature at every time step
            X_seq = np.concatenate([X_3d, T_3d], axis=2)
            y_target = Y_2d[:, -1]  # predict final-period outcome

            self.treatment_values = np.unique(T)
            self.seq_len_ = n_periods
            self.model = clone(self.base_model)
            self.model.fit(X_seq, y_target)

            # Save patient-level training units for clustered bootstrap CI.
            self._ci_train_mode = "seq"
            self._ci_train_X_seq_with_t = X_seq.astype(np.float32, copy=True)
            self._ci_train_y = y_target.astype(np.float64, copy=True)
            self._bootstrap_cache = {}
            return self

        return self.fit(X, T, Y)
    



