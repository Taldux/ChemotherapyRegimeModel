"""
@file causal_forest.py
@brief Causal Forest CATE estimator wrapping econml's CausalForestDML.
"""

import numpy as np
from typing import Any, cast
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from econml.dml import CausalForestDML

from src.models.panel_utils import balance_panel


class CausalForest(BaseEstimator):
    """
    @brief Wrapper around econml's CausalForestDML for multi-treatment CATE estimation.

    @param n_estimators      Number of trees in the causal forest.
    @param max_depth         Maximum depth of trees (None = unlimited).
    @param min_samples_leaf  Minimum samples per leaf.
    @param random_state      Random seed.
    """
    
    def __init__(self, n_estimators=200, max_depth=None, min_samples_leaf=10,
                 random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
    
    def _build_model(self):
        """@brief Create a fresh CausalForestDML instance."""
        return CausalForestDML(
            model_y=cast(Any, RandomForestRegressor(
                n_estimators=100, max_depth=10,
                min_samples_leaf=5, random_state=self.random_state, n_jobs=-1
            )),
            model_t=cast(Any, RandomForestClassifier(
                n_estimators=100, max_depth=5,
                random_state=self.random_state, n_jobs=-1
            )),
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            discrete_treatment=True,
            inference=True,
        )

    def fit(self, X, t, y):
        """
        @brief Fit the Causal Forest on cross-sectional data.

        @param X  Feature matrix of shape (n_samples, n_features).
        @param t  Treatment vector of shape (n_samples,) in {0, ..., K-1}.
        @param y  Outcome vector of shape (n_samples,).
        @return self
        """
        X = np.array(X)
        t = np.array(t).ravel()
        y = np.array(y).ravel()
        
        self.treatment_values = np.sort(np.unique(t))
        self.n_treatments = len(self.treatment_values)
        self.reference_treatment = self.treatment_values[0]
        
        self.model_ = self._build_model()
        self.model_.fit(Y=y, T=t, X=X)
        
        self._fit_baseline_model(X, t, y)
        
        return self

    def fit_panel(self, Y, T, X, groups, max_periods=None):
        """
        @brief Fit on longitudinal panel data with patient-grouped cross-fitting.

        Balances panels, then fits CausalForestDML ensuring all observations
        from the same patient stay in the same CV fold.

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

        self.treatment_values = np.sort(np.unique(T))
        self.n_treatments = len(self.treatment_values)
        self.reference_treatment = self.treatment_values[0]

        self.model_ = self._build_model()
        self.model_.fit(Y=Y, T=T, X=X, groups=groups)

        # Baseline model on first-period observations (one per patient)
        assert X is not None
        unique_groups, first_indices = np.unique(groups, return_index=True)
        self._fit_baseline_model(
            X[first_indices], T[first_indices], Y[first_indices]
        )

        return self

    def _fit_baseline_model(self, X, t, y):
        """
        @brief Fit E[Y | X, T=reference] for absolute outcome reconstruction.

        @param X  Feature matrix.
        @param t  Treatment vector.
        @param y  Outcome vector.
        """
        mask_ref = (t == self.reference_treatment)
        self.baseline_model_ = RandomForestRegressor(
            n_estimators=100, max_depth=10,
            min_samples_leaf=5, random_state=self.random_state, n_jobs=-1
        )
        self.baseline_model_.fit(X[mask_ref], y[mask_ref])
    
    def _get_cate(self, X, treatment):
        """
        @brief Retrieve CATE for a treatment versus the reference arm.

        @param X          Feature matrix.
        @param treatment  Treatment value.
        @return CATE estimates of shape (n_samples,).
        """
        X = np.asarray(X, dtype=float)
        if treatment == self.reference_treatment:
            return np.zeros(X.shape[0])

        return self.model_.effect(
            X=X,
            T0=self.reference_treatment,
            T1=treatment,
        ).ravel()
    
    def predict(self, X, treatment_from, treatment_to):
        """
        @brief Predict ITE for switching from treatment_from to treatment_to.

        @param X               Feature matrix of shape (n_samples, n_features).
        @param treatment_from  Source treatment value.
        @param treatment_to    Target treatment value.
        @return ITE estimates of shape (n_samples,).
        """
        X = np.asarray(X, dtype=float)

        if treatment_from == treatment_to:
            return np.zeros(X.shape[0])

        return self.model_.effect(
            X=X,
            T0=treatment_from,
            T1=treatment_to,
        ).ravel()
    
    def predict_outcome(self, X, treatment):
        """
        @brief Predict E[Y(treatment) | X] = mu_ref(X) + CATE(ref -> treatment).

        @param X          Feature matrix of shape (n_samples, n_features).
        @param treatment  Treatment value.
        @return Predicted outcomes of shape (n_samples,).
        """
        X = np.asarray(X, dtype=float)
        
        mu_ref = self.baseline_model_.predict(X)
        cate = self._get_cate(X, treatment)
        
        return mu_ref + cate
    
    def predict_effect_with_ci(self, X, treatment_from, treatment_to, alpha=0.05):
        """
        @brief Predict treatment effect with confidence intervals.

        @param X               Feature matrix of shape (n_samples, n_features).
        @param treatment_from  Source treatment value.
        @param treatment_to    Target treatment value.
        @param alpha           Significance level (default 0.05 for 95% CI).
        @return Dict with keys 'effect', 'ci_lower', 'ci_upper' each of shape (n_samples,).
        """
        X = np.asarray(X, dtype=float)

        if treatment_to == self.reference_treatment and treatment_from == self.reference_treatment:
            n = X.shape[0]
            return {
                'effect': np.zeros(n),
                'ci_lower': np.zeros(n),
                'ci_upper': np.zeros(n)
            }

        effect = self.model_.effect(
            X=X,
            T0=treatment_from,
            T1=treatment_to,
        ).ravel()
        inference = self.model_.effect_inference(
            X=X,
            T0=treatment_from,
            T1=treatment_to,
        )
        ci_lower, ci_upper = inference.conf_int(alpha=alpha)
        
        return {
            'effect': effect,
            'ci_lower': np.asarray(ci_lower).ravel(),
            'ci_upper': np.asarray(ci_upper).ravel()
        }
