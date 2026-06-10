"""
@file tlearner.py
@brief T-Learner causal effect estimator using separate per-arm models.
"""

import numpy as np
from sklearn.base import BaseEstimator, clone

from src.models.panel_utils import balance_panel


class TLearner(BaseEstimator):
    """
    @brief T-Learner: separate arm models for CATE estimation.

    Fits one regression model per treatment arm; predicts counterfactuals
    from arm-specific models.

    @param base_model  sklearn-compatible regression estimator.
    """
    
    def __init__(self, base_model):

        self.base_model = base_model
        self._bootstrap_cache = {}
        
    def fit(self, X, t, y):
        """
        @brief Fit one model per treatment arm on (X -> Y) subsets.

        @param X  Feature matrix of shape (n_samples, n_features).
        @param t  Treatment vector of shape (n_samples,).
        @param y  Outcome vector of shape (n_samples,).
        @return self
        """
        X = np.asarray(X)
        t = np.asarray(t).ravel()
        y = np.asarray(y).ravel()
        
        self.treatment_values = np.unique(t)
        
        # fit a model for each treatment arm
        self.models = {}
        for treatment_val in self.treatment_values:
            mask = (t == treatment_val)
            if not mask.any():
                raise ValueError(f"No samples found for treatment {treatment_val}")
            
            X_t = X[mask]
            y_t = y[mask]
            
            model = clone(self.base_model)
            model.fit(X_t, y_t)
            self.models[treatment_val] = model

        # Save training data for bootstrap CI.
        self._ci_train_mode = "flat"
        self._ci_train_X = X.astype(np.float64, copy=True)
        self._ci_train_t = t.astype(int, copy=True)
        self._ci_train_y = y.astype(np.float64, copy=True)
        self._bootstrap_cache = {}
        
        return self
    
    def predict(self, X, treatment_from, treatment_to):
        """
        @brief Predict ITE of switching between two treatments.

        @param X               Feature matrix of shape (n_samples, n_features).
        @param treatment_from  Source treatment value.
        @param treatment_to    Target treatment value.
        @return ITE estimates of shape (n_samples,).
        """
        X = np.array(X)
        
        if treatment_from not in self.models:
            raise ValueError(f"Treatment {treatment_from} not seen during training")
        if treatment_to not in self.models:
            raise ValueError(f"Treatment {treatment_to} not seen during training")
        
        mu_from = self.predict_outcome(X, treatment_from)
        mu_to   = self.predict_outcome(X, treatment_to)
        return mu_to - mu_from
    
    def predict_outcome(self, X, treatment):
        """
        @brief Predict E[Y(treatment) | X] using the arm-specific model.

        @param X          Feature matrix of shape (n_samples, n_features).
        @param treatment  Treatment value.
        @return Predicted outcomes of shape (n_samples,).
        """
        X = np.array(X)

        if treatment not in self.models:
            raise ValueError(f"Treatment {treatment} not seen during training")

        # For sequential models: use stored evaluation sequences if available
        if (getattr(self.base_model, '_is_sequential', False)
                and hasattr(self, '_eval_X_seq')
                and len(X) == len(self._eval_X_seq)):
            return self.models[treatment].predict(self._eval_X_seq)

        return self.models[treatment].predict(X)

    def set_eval_sequences(self, X_seq):
        """
        @brief Store 3-D evaluation sequences for LSTM prediction.

        @param X_seq  Array of shape (n_eval, max_periods, n_features).
        """
        self._eval_X_seq = np.asarray(X_seq, dtype=np.float32)

    def predict_effect_with_ci(self, X, from_treatment, to_treatment,
                                n_mc_samples=50, alpha=0.05,
                                ci_method="bootstrap",
                                n_bootstrap=100,
                                random_state=42):
        """
        @brief Predict ITE with confidence intervals.

        Primary path: arm-stratified non-parametric bootstrap over training
        units (rows for flat mode, patients for panel/LSTM mode). Fallback:
        tree-variance (RF) or MC dropout (LSTM).

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

        if from_treatment not in self.models or to_treatment not in self.models:
            raise ValueError("Treatment not seen during training.")

        m_from = self.models[from_treatment]
        m_to   = self.models[to_treatment]

        if hasattr(m_from, 'estimators_') and hasattr(m_to, 'estimators_'):
            # ── Random Forest: use per-tree variance ──────────────────────────
            X = np.asarray(X)
            tree_from = np.array([t.predict(X) for t in m_from.estimators_])
            tree_to   = np.array([t.predict(X) for t in m_to.estimators_])
            # Propagate variance: Var(Y1 - Y0) ≈ Var(Y1) + Var(Y0)
            effect = tree_to.mean(axis=0) - tree_from.mean(axis=0)
            std    = np.sqrt(tree_from.var(axis=0) + tree_to.var(axis=0))

        elif (getattr(m_from, '_is_sequential', False)
              and hasattr(m_from, 'predict_with_uncertainty')):
            # ── LSTM: MC Dropout ──────────────────────────────────────────────
            X_seq = np.asarray(X, dtype=np.float32)  # (n, seq_len, n_feat)
            mu_from, sig_from = m_from.predict_with_uncertainty(
                X_seq, n_samples=n_mc_samples)
            mu_to,   sig_to   = m_to.predict_with_uncertainty(
                X_seq, n_samples=n_mc_samples)
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
            point_preds = {t: self.models[t].predict(X_eval) for t in tr_vals}
        else:
            X_eval = np.asarray(X, dtype=np.float64)
            if X_eval.ndim != 2:
                return None
            point_preds = {t: self.models[t].predict(X_eval) for t in tr_vals}

        rng = np.random.default_rng(random_state)
        n_eval = len(X_eval)
        boot_outcomes = {
            t: np.empty((n_bootstrap, n_eval), dtype=np.float64)
            for t in tr_vals
        }

        X_train = self._ci_train_X
        t_train = self._ci_train_t
        y_train = self._ci_train_y
        if len(y_train) == 0:
            return None

        arm_indices = {t: np.where(t_train == t)[0] for t in tr_vals}
        if any(len(idx) == 0 for idx in arm_indices.values()):
            return None

        for b in range(n_bootstrap):
            bs_models = {}
            for t in tr_vals:
                arm_idx = arm_indices[t]
                draw_idx = rng.choice(arm_idx, size=len(arm_idx), replace=True)
                m = clone(self.base_model)
                m.fit(X_train[draw_idx], y_train[draw_idx])
                bs_models[t] = m

            for t in tr_vals:
                boot_outcomes[t][b] = bs_models[t].predict(X_eval)

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

    def fit_panel(self, Y, T, X, groups, max_periods=None):
        """
        @brief Fit on longitudinal panel data.

        Balances panels then delegates to fit(). For LSTM base models,
        reshapes into patient-level 3-D sequences.

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
            T_2d = T.reshape(n_groups, n_periods)
            Y_2d = Y.reshape(n_groups, n_periods)

            # Patient-level treatment = first-period treatment
            T_patient = T_2d[:, 0]
            y_target = Y_2d[:, -1]  # final-period outcome (not mean)

            self.treatment_values = np.unique(T_patient)
            self.seq_len_ = n_periods
            self.models = {}
            for treatment_val in self.treatment_values:
                mask = (T_patient == treatment_val)
                if not mask.any():
                    raise ValueError(f"No patients for treatment {treatment_val}")
                model = clone(self.base_model)
                model.fit(X_3d[mask], y_target[mask])
                self.models[treatment_val] = model

            # Save patient-level training units for clustered bootstrap CI.
            self._ci_train_mode = "seq"
            self._ci_train_X = X_3d.astype(np.float32, copy=True)
            self._ci_train_t = T_patient.astype(int, copy=True)
            self._ci_train_y = y_target.astype(np.float64, copy=True)
            self._bootstrap_cache = {}
            return self

        return self.fit(X, T, Y)

