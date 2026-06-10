"""
@file bowl.py
@brief Backwards Outcome Weighted Learning (BOWL) for optimal treatment regime estimation.
"""

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from src.models.panel_utils import balance_panel


class BOWL(BaseEstimator):
    """
    @brief BOWL policy learner using outcome-weighted pairwise classification.

    For K treatments, trains K*(K-1)/2 pairwise classifiers; optimal
    treatment is chosen by plurality voting.

    @param base_classifier   sklearn classifier supporting sample_weight.
    @param propensity_model  Propensity estimator; None uses empirical frequencies.
    @param C                 Regularisation parameter for the default SVM.
    @param kernel            Kernel type for the default SVM.
    @param scale_outcomes    Scale outcomes to [0, 1] for numerical stability.
    """
    
    def __init__(self, base_classifier=None, propensity_model=None, 
                 C=1.0, kernel='rbf', scale_outcomes=True):
        self.base_classifier = base_classifier
        self.propensity_model = propensity_model
        self.C = C
        self.kernel = kernel
        self.scale_outcomes = scale_outcomes
        
    def _get_classifier(self):
        """@brief Return a fresh classifier instance."""
        if self.base_classifier is not None:
            return clone(self.base_classifier)
        # Default: SVM with RBF kernel (common choice for OWL)
        return SVC(C=self.C, kernel=self.kernel, probability=True)
    
    def _estimate_propensities(self, t):
        """
        @brief Estimate marginal treatment probabilities from observed assignments.

        @param t  Treatment vector of shape (n_samples,).
        @return Dict mapping treatment value to empirical frequency.
        """
        t = np.array(t).ravel()
        unique, counts = np.unique(t, return_counts=True)
        props = {treatment: count / len(t) for treatment, count in zip(unique, counts)}
        return props
    
    def _scale_outcomes(self, y):
        """
        @brief Scale outcomes to [0, 1] for numerical stability.

        @param y  Outcome vector of shape (n_samples,).
        @return Tuple (y_scaled, (y_min, y_max)).
        """
        y = np.array(y).ravel()
        y_min, y_max = y.min(), y.max()
        if y_max - y_min < 1e-10:
            return y, (0, 1)
        y_scaled = (y - y_min) / (y_max - y_min)
        return y_scaled, (y_min, y_max)
    
    def fit(self, X, t, y, propensity_scores=None):
        """
        @brief Fit pairwise outcome-weighted classifiers.

        @param X                  Feature matrix of shape (n_samples, n_features).
        @param t                  Treatment vector of shape (n_samples,).
        @param y                  Outcome vector of shape (n_samples,).
        @param propensity_scores  Per-sample propensity weights; estimated if None.
        @return self
        """
        X = np.array(X)
        t = np.array(t).ravel()
        y = np.array(y).ravel()
        
        # Store treatment values
        self.treatment_values_ = np.unique(t)
        n_treatments = len(self.treatment_values_)
        
        # Scale outcomes for numerical stability
        if self.scale_outcomes:
            y_scaled, self.outcome_scaler_ = self._scale_outcomes(y)
        else:
            y_scaled = y
            self.outcome_scaler_ = None
        
        # Estimate propensities if not provided
        if propensity_scores is None:
            self.propensities_ = self._estimate_propensities(t)
            propensity_scores = np.array([self.propensities_[ti] for ti in t])
        else:
            propensity_scores = np.array(propensity_scores)
            self.propensities_ = self._estimate_propensities(t)
        
        # Train pairwise classifiers
        self.classifiers_ = {}
        self.scalers_ = {}
        
        for i, t1 in enumerate(self.treatment_values_):
            for t2 in self.treatment_values_[i+1:]:
                # Get samples from both treatments
                mask1 = (t == t1)
                mask2 = (t == t2)
                mask = mask1 | mask2
                
                if mask.sum() < 10:  # Need minimum samples
                    continue
                
                X_pair = X[mask]
                t_pair = t[mask]
                y_pair = y_scaled[mask]
                prop_pair = propensity_scores[mask]
                
                # Create binary labels: 1 for t2, 0 for t1
                labels = (t_pair == t2).astype(int)
                
                # Weight = Y / propensity (Y already non-negative after scaling)
                weights = y_pair / np.maximum(prop_pair, 1e-6)
                
                # Scale features for SVM
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_pair)
                
                # Fit weighted classifier
                clf = self._get_classifier()
                clf.fit(X_scaled, labels, sample_weight=weights)
                
                # Store classifier and scaler
                self.classifiers_[(t1, t2)] = clf
                self.scalers_[(t1, t2)] = scaler
        
        return self

    def fit_panel(self, Y, T, X, groups, max_periods=None):
        """
        @brief Fit on longitudinal panel data.

        Balances panels then delegates to fit().

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

        return self.fit(X, T, Y)

    def _predict_pairwise(self, X, t1, t2):
        """
        @brief Predict preferred treatment between t1 and t2.

        @param X   Feature matrix.
        @param t1  First treatment value.
        @param t2  Second treatment value.
        @return Binary array (0 = prefer t1, 1 = prefer t2).
        """
        X = np.array(X)
        
        # Find the classifier for this pair (may be stored in either order)
        if (t1, t2) in self.classifiers_:
            clf = self.classifiers_[(t1, t2)]
            scaler = self.scalers_[(t1, t2)]
            flip = False
        elif (t2, t1) in self.classifiers_:
            clf = self.classifiers_[(t2, t1)]
            scaler = self.scalers_[(t2, t1)]
            flip = True
        else:
            raise ValueError(f"No classifier found for treatments {t1} and {t2}")
        
        X_scaled = scaler.transform(X)
        preds = clf.predict(X_scaled)
        
        if flip:
            preds = 1 - preds
            
        return preds
    
    def predict_optimal_treatment(self, X):
        """
        @brief Predict the optimal treatment for each sample via plurality voting.

        @param X  Feature matrix of shape (n_samples, n_features).
        @return Array of optimal treatment values of shape (n_samples,).
        """
        X = np.array(X)
        n_samples = X.shape[0]
        n_treatments = len(self.treatment_values_)
        
        # Count votes for each treatment
        votes = np.zeros((n_samples, n_treatments))
        
        for i, t1 in enumerate(self.treatment_values_):
            for j, t2 in enumerate(self.treatment_values_[i+1:], i+1):
                preds = self._predict_pairwise(X, t1, t2)
                # preds=0 means t1 wins, preds=1 means t2 wins
                votes[:, i] += (preds == 0)
                votes[:, j] += (preds == 1)
        
        # Return treatment with most votes
        best_idx = np.argmax(votes, axis=1)
        return np.array([self.treatment_values_[idx] for idx in best_idx])
    
    def predict_treatment_probability(self, X, treatment):
        """
        @brief Estimate the win-rate probability that a treatment is optimal for each sample.

        @param X          Feature matrix of shape (n_samples, n_features).
        @param treatment  Treatment value to evaluate.
        @return Win-rate estimates of shape (n_samples,).
        """
        X = np.array(X)
        n_samples = X.shape[0]
        
        if treatment not in self.treatment_values_:
            raise ValueError(f"Treatment {treatment} not seen during training")
        
        t_idx = np.where(self.treatment_values_ == treatment)[0][0]
        n_comparisons = len(self.treatment_values_) - 1
        
        if n_comparisons == 0:
            return np.ones(n_samples)
        
        # Count wins for this treatment
        wins = np.zeros(n_samples)
        
        for i, other_t in enumerate(self.treatment_values_):
            if other_t == treatment:
                continue
            
            try:
                preds = self._predict_pairwise(X, treatment, other_t)
                wins += (preds == 0)  # treatment wins when pred=0
            except ValueError:
                continue
        
        return wins / n_comparisons
    
    def predict(self, X, treatment_from, treatment_to):
        """
        @brief Predict preference score for switching treatments.

        @param X               Feature matrix of shape (n_samples, n_features).
        @param treatment_from  Source treatment value.
        @param treatment_to    Target treatment value.
        @return Scores of shape (n_samples,): +1 prefers to, -1 prefers from.
        """
        preds = self._predict_pairwise(X, treatment_from, treatment_to)
        # Convert to preference score: 0 -> -1 (prefer from), 1 -> +1 (prefer to)
        return 2 * preds - 1
    
    def evaluate_policy(self, X, Y_true_dict):
        """
        @brief Evaluate the learned policy against ground-truth potential outcomes.

        @param X            Feature matrix of shape (n_samples, n_features).
        @param Y_true_dict  Dict mapping treatment index to true outcomes array.
        @return Dict with policy evaluation metrics.
        """
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Get optimal treatments from policy
        recommended = self.predict_optimal_treatment(X)
        
        # Calculate value of recommended policy
        policy_outcomes = np.array([Y_true_dict[recommended[i]][i] 
                                   for i in range(n_samples)])
        
        # Calculate value of oracle policy (always picks best)
        oracle_treatments = []
        oracle_outcomes = []
        for i in range(n_samples):
            best_t = max(Y_true_dict.keys(), 
                        key=lambda t: Y_true_dict[t][i])
            oracle_treatments.append(best_t)
            oracle_outcomes.append(Y_true_dict[best_t][i])
        oracle_outcomes = np.array(oracle_outcomes)
        
        # Calculate accuracy (how often we pick the true best)
        accuracy = np.mean([recommended[i] == oracle_treatments[i] 
                           for i in range(n_samples)])
        
        # Calculate regret (difference from oracle)
        regret = np.mean(oracle_outcomes - policy_outcomes)
        
        # Value function (expected outcome under policy)
        value = np.mean(policy_outcomes)
        oracle_value = np.mean(oracle_outcomes)
        
        return {
            'accuracy': accuracy,
            'regret': regret,
            'value': value,
            'oracle_value': oracle_value,
            'relative_value': value / oracle_value if oracle_value != 0 else float('inf')
        }



