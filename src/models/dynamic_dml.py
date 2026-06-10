"""
@file dynamic_dml.py
@brief Dynamic Double Machine Learning (DynamicDML) CATE estimator for panel data.
"""

from typing import Any, cast

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from econml.panel.dml import DynamicDML as _EconDynamicDML

from src.models.panel_utils import balance_panel


class DynamicDMLModel(BaseEstimator):
    """
    @brief Wrapper around econml's DynamicDML for multi-treatment CATE estimation.

    Supports panel/longitudinal fitting (fit_panel) and cross-sectional
    fitting (fit). Public API mirrors CausalForest for drop-in compatibility.

    @param n_estimators      Number of trees in first-stage nuisance models.
    @param max_depth         Maximum depth of first-stage Random Forest trees.
    @param min_samples_leaf  Minimum samples per leaf in nuisance models.
    @param cv                Cross-validation folds for nuisance estimation.
    @param random_state      Random seed.
    """

    def __init__(self, n_estimators=100, max_depth=10, min_samples_leaf=5,
                 cv=2, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.cv = cv
        self.random_state = random_state

    @staticmethod
    def _safe_cv(T, groups, requested_cv):
        """
        @brief Cap CV folds to the minimum number of distinct groups per treatment.

        @param T             Treatment vector.
        @param groups        Group identifiers.
        @param requested_cv  Requested number of CV folds.
        @return CV fold count (at least 2).
        """
        min_groups = min(
            len(np.unique(groups[T == t_val]))
            for t_val in np.unique(T)
        )
        n_splits = max(2, min(requested_cv, min_groups))
        if n_splits < requested_cv:
            print(f"  DynamicDML: capping CV folds from {requested_cv} to "
                  f"{n_splits} (rarest treatment has {min_groups} groups).")
        return n_splits

    @staticmethod
    def _first_period_indices(groups):
        """
        @brief Return row indices of the first observation for each group.

        @param groups  Group identifier array.
        @return Integer index array of shape (n_groups,).
        """
        _, first_indices = np.unique(groups, return_index=True)
        return first_indices

    @staticmethod
    def _first_period_missing_treatments(T, groups):
        """
        @brief Return the first balanced period that is missing a treatment arm.

        @param T       Treatment vector.
        @param groups  Group identifiers (must be balanced to equal length).
        @return Tuple (period_idx, missing_treatments), or (None, None) if all complete.
        """
        unique_groups, group_sizes = np.unique(groups, return_counts=True)
        unique_sizes = np.unique(group_sizes)
        if len(unique_sizes) != 1:
            return None, None

        n_groups = len(unique_groups)
        n_periods = unique_sizes[0]
        all_treatments = np.sort(np.unique(T))
        T_panel = T.reshape(n_groups, n_periods)

        for period_idx in range(n_periods):
            seen = np.unique(T_panel[:, period_idx])
            missing = np.setdiff1d(all_treatments, seen)
            if len(missing) > 0:
                return period_idx, missing

        return None, None

    def _prepare_cate_features(self, X, groups=None, fit=False):
        """
        @brief Drop constant first-period features before passing X to DynamicDML.

        @param X       Feature matrix or None.
        @param groups  Group identifiers (required when fit=True).
        @param fit     If True, compute and store the feature mask.
        @return Filtered feature matrix, or None if all features are constant.
        """
        if X is None:
            if fit:
                self.cate_feature_mask_ = None
                self.uses_cate_features_ = False
            return None

        X = np.asarray(X, dtype=float)

        if fit:
            X_first = X[self._first_period_indices(groups)] if groups is not None else X
            mask = np.nanstd(X_first, axis=0) > 1e-12
            dropped = int(mask.size - mask.sum())
            if dropped > 0:
                print(
                    "  DynamicDML: dropping "
                    f"{dropped} constant first-period feature(s) from the final stage."
                )
            self.cate_feature_mask_ = mask
            self.uses_cate_features_ = bool(np.any(mask))
            if not self.uses_cate_features_:
                print(
                    "  DynamicDML: all first-period features are constant; "
                    "estimating marginal effects without heterogeneity features."
                )
                return None

        if not getattr(self, 'uses_cate_features_', False):
            return None

        return X[:, self.cate_feature_mask_]

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X, t, y):
        """
        @brief Fit on cross-sectional data (each sample = single-period panel).

        @param X  Feature matrix of shape (n_samples, n_features).
        @param t  Treatment vector of shape (n_samples,).
        @param y  Outcome vector of shape (n_samples,).
        @return self
        """
        X = np.asarray(X, dtype=float)
        t = np.asarray(t).ravel()
        y = np.asarray(y, dtype=float).ravel()

        self.treatment_values_ = np.sort(np.unique(t))
        self.n_treatments_ = len(self.treatment_values_)
        self.reference_treatment_ = self.treatment_values_[0]
        self.fit_mode_ = "cross-sectional"

        # Every sample is its own group with n_periods = 1
        groups = np.arange(len(y))
        X_cate = self._prepare_cate_features(X, groups=groups, fit=True)

        self.model_ = _EconDynamicDML(
            model_y=cast(Any, RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
            )),
            model_t=cast(Any, RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=5,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
            )),
            discrete_treatment=True,
            cv=self._safe_cv(t, groups, self.cv),
            random_state=self.random_state,
        )

        self.model_.fit(Y=y, T=t, X=X_cate, groups=groups)

        # Baseline outcome model for absolute outcome prediction
        self._fit_baseline_model(X, t, y)
        return self

    def fit_panel(self, Y, T, X, groups, W=None, max_periods=None):
        """
        @brief Fit on panel/longitudinal data.

        Balances unequal-length panels (LOCF-pad short, truncate long),
        checks within-group treatment variation, then fits DynamicDML.
        Falls back to cross-sectional fit() on singular matrix errors.

        @param Y            Outcomes of shape (n_total,).
        @param T            Treatments of shape (n_total,).
        @param X            Features of shape (n_total, d_x), or None.
        @param groups       Group (patient) IDs of shape (n_total,).
        @param W            Time-varying confounders of shape (n_total, d_w), or None.
        @param max_periods  Target periods per group; defaults to median group size.
        @return self
        """
        Y = np.asarray(Y, dtype=float).ravel()
        T = np.asarray(T).ravel()
        X = np.asarray(X, dtype=float) if X is not None else None
        W = np.asarray(W, dtype=float) if W is not None else None
        groups = np.asarray(groups)

        # --- Re-group by patient when treatment is constant within groups ---
        # Semi-time data encodes groups as pid*10 + treatment (constant
        # treatment per group).  DynamicDML requires within-group treatment
        # variation, so we collapse to patient-level groups (groups // 10)
        # and re-sort contiguously.
        if not self._has_within_group_variation(T, groups):
            print("  DynamicDML: constant treatment per group detected "
                  ", re-grouping by patient (groups // 10).")
            groups = groups // 10
            sort_idx = np.argsort(groups, kind='stable')
            Y, T, groups = Y[sort_idx], T[sort_idx], groups[sort_idx]
            if X is not None:
                X = X[sort_idx]
            if W is not None:
                W = W[sort_idx]

        # --- Balance panels ---
        Y, T, X, groups, W = balance_panel(
            Y, T, X, groups, W, max_periods=max_periods
        )

        bad_period, missing_treatments = self._first_period_missing_treatments(T, groups)
        if bad_period is not None:
            missing_values = [] if missing_treatments is None else missing_treatments
            missing_str = ", ".join(str(int(t_val)) for t_val in missing_values)
            raise ValueError(
                "DynamicDML requires every balanced period to contain all treatments "
                f"across groups, but period {bad_period} is missing treatment(s) "
                f"{missing_str}. This usually means the panel contains "
                "counterfactual-expanded trajectories instead of one observed "
                "treatment path per patient."
            )

        self.treatment_values_ = np.sort(np.unique(T))
        self.n_treatments_ = len(self.treatment_values_)
        self.reference_treatment_ = self.treatment_values_[0]
        X_cate = self._prepare_cate_features(X, groups=groups, fit=True)
        self.fit_mode_ = "panel"

        self.model_ = _EconDynamicDML(
            model_y=cast(Any, RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
            )),
            model_t=cast(Any, RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=5,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
            )),
            discrete_treatment=True,
            cv=self._safe_cv(T, groups, self.cv),
            random_state=self.random_state,
        )
        unique_groups_after = np.unique(groups)
        self.max_periods_ = int(len(Y) // len(unique_groups_after))
        try:
            self.model_.fit(Y=Y, T=T, X=X_cate, W=W, groups=groups)
        except (np.linalg.LinAlgError, AttributeError) as e:
            print(
                f"DynamicDML: {type(e).__name__}, falling back to "
                "cross-sectional DynamicDML fit on first-period data."
            )
            first_idx = self._first_period_indices(groups)
            return self.fit(
                X[first_idx] if X is not None else X,
                T[first_idx],
                Y[first_idx],
            )

        # Baseline outcome model on first-period observations
        first_indices = self._first_period_indices(groups)
        X_first = X[first_indices] if X is not None else None
        T_first = T[first_indices]
        Y_first = Y[first_indices]

        if X_first is not None:
            self._fit_baseline_model(X_first, T_first, Y_first)

        return self

    @staticmethod
    def _has_within_group_variation(T, groups):
        """
        @brief Check whether at least one group contains more than one treatment.

        @param T       Treatment vector.
        @param groups  Group identifiers.
        @return True if any of the first 50 groups has >1 unique treatment.
        """
        for g in np.unique(groups)[:50]:
            if len(np.unique(T[groups == g])) > 1:
                return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_baseline_model(self, X, t, y):
        """
        @brief Fit E[Y | X, T=reference] for absolute outcome reconstruction.

        @param X  Feature matrix.
        @param t  Treatment vector.
        @param y  Outcome vector.
        """
        mask_ref = t == self.reference_treatment_
        self.baseline_model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=1,  # single-row eval calls; threading overhead kills throughput
        )
        if mask_ref.sum() > 0:
            self.baseline_model_.fit(X[mask_ref], y[mask_ref])
        else:
            # Fallback: if reference arm is somehow empty
            self.baseline_model_.fit(X, y)

    # ------------------------------------------------------------------
    # Prediction – same signatures as CausalForest
    # ------------------------------------------------------------------
    
    def predict(self, X, treatment_from, treatment_to):
        """
        @brief Predict CATE for switching from treatment_from to treatment_to.

        @param X               Feature matrix of shape (n, d).
        @param treatment_from  Source treatment code.
        @param treatment_to    Target treatment code.
        @return Effect estimates of shape (n,).
        """
        X = np.asarray(X, dtype=float)
        if treatment_from == treatment_to:
            return np.zeros(X.shape[0])
        X_cate = self._prepare_cate_features(X)
        return self.model_.effect(X=X_cate, T0=treatment_from, T1=treatment_to).ravel()

    def predict_outcome(self, X, treatment):
        """
        @brief Predict E[Y(treatment) | X] = E[Y(ref) | X] + CATE(ref -> treatment).

        @param X          Feature matrix of shape (n, d).
        @param treatment  Treatment code.
        @return Outcome estimates of shape (n,).
        """
        X = np.asarray(X, dtype=float)
        mu_ref = self.baseline_model_.predict(X)
        if treatment == self.reference_treatment_:
            return mu_ref
        X_cate = self._prepare_cate_features(X)
        cate = self.model_.effect(
            X=X_cate,
            T0=self.reference_treatment_,
            T1=treatment,
        ).ravel()
        return mu_ref + cate

    def predict_outcome_sequential(self, features_so_far, past_treatments, treatment):
        """
        @brief Predict expected outcome under `treatment` at the current visit
               using the patient's observed treatment history.

        Constructs a length-max_periods_ treatment trajectory from
        past_treatments, places `treatment` at the current step, and fills any
        remaining future steps with `treatment` (continuation assumption). The
        baseline trajectory uses the reference treatment from the current step
        onward. Returns mu_ref(X_first) + effect over the full panel, where
        X_first is the first visit's feature row.

        @param features_so_far  Array of shape (n_steps_so_far, n_features).
                                Used to extract the first-period covariates.
        @param past_treatments  Array of shape (n_steps_so_far - 1,) with the
                                treatments given before the current visit.
        @param treatment        Candidate treatment code for the current visit.
        @return Predicted cumulative outcome scalar.
        """
        if not hasattr(self, "max_periods_"):
            raise RuntimeError(
                "predict_outcome_sequential requires fit_panel to have been "
                "called so that max_periods_ is available."
            )

        features_so_far = np.asarray(features_so_far, dtype=float)
        past = np.asarray(past_treatments, dtype=int).ravel()
        max_periods = self.max_periods_

        X_first = features_so_far[0:1]

        current_step = len(past)
        if current_step >= max_periods:
            past = past[-(max_periods - 1):]
            current_step = max_periods - 1

        T1 = np.concatenate([
            past,
            np.full(max_periods - current_step, treatment, dtype=int),
        ])
        T0 = np.concatenate([
            past,
            np.full(max_periods - current_step, self.reference_treatment_, dtype=int),
        ])

        mu_ref = float(self.baseline_model_.predict(X_first)[0])

        if np.array_equal(T1, T0):
            return mu_ref

        X_cate = self._prepare_cate_features(X_first)
        effect = float(self.model_.effect(
            X=X_cate,
            T0=T0.reshape(1, -1),
            T1=T1.reshape(1, -1),
        ).ravel()[0])
        return mu_ref + effect

    def predict_effect_with_ci(self, X, treatment_from, treatment_to, alpha=0.05):
        """
        @brief Predict CATE with confidence intervals.

        @param X               Feature matrix of shape (n, d).
        @param treatment_from  Source treatment code.
        @param treatment_to    Target treatment code.
        @param alpha           Significance level (default 0.05 for 95% CI).
        @return Dict with keys 'effect', 'ci_lower', 'ci_upper' each of shape (n,).
        """
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        if treatment_from == treatment_to:
            return {
                "effect": np.zeros(n),
                "ci_lower": np.zeros(n),
                "ci_upper": np.zeros(n),
            }

        X_cate = self._prepare_cate_features(X)
        effect = self.model_.effect(X_cate, T0=treatment_from, T1=treatment_to).ravel()
        inference = self.model_.effect_inference(X_cate, T0=treatment_from, T1=treatment_to)
        ci_lower, ci_upper = inference.conf_int(alpha=alpha)

        return {
            "effect": effect,
            "ci_lower": np.asarray(ci_lower).ravel(),
            "ci_upper": np.asarray(ci_upper).ravel(),
        }
