"""
@file metrics.py
@brief Causal model evaluation metrics and policy evaluation utilities.
"""

import numpy as np


def calculate_pehe(tau_pred, tau_true):
    """
    @brief Compute Precision in Estimation of Heterogeneous Effects (PEHE).

    @param tau_pred  Predicted ITE array of shape (n,).
    @param tau_true  True ITE array of shape (n,).
    @return sqrt(mean((tau_pred - tau_true)^2)) as a float.
    """
    tau_pred = np.array(tau_pred).ravel()
    tau_true = np.array(tau_true).ravel()
    
    return np.sqrt(np.mean((tau_pred - tau_true) ** 2))


def calculate_ate_error(tau_pred, tau_true):
    """
    @brief Compute absolute error in the Average Treatment Effect (ATE).

    @param tau_pred  Predicted ITE array of shape (n,).
    @param tau_true  True ITE array of shape (n,).
    @return |mean(tau_pred) - mean(tau_true)| as a float.
    """
    ate_pred = np.mean(tau_pred)
    ate_true = np.mean(tau_true)
    
    return abs(ate_pred - ate_true)


def evaluate_semi_synthetic(tlearner, X, Y_true_dict, treatment_from, treatment_to):
    """
    @brief Evaluate a causal model on semi-synthetic potential outcomes.

    @param tlearner        Fitted model with a predict() method.
    @param X               Feature matrix of shape (n_samples, n_features).
    @param Y_true_dict     Dict mapping treatment index to true outcomes array.
    @param treatment_from  Source treatment index.
    @param treatment_to    Target treatment index.
    @return Dict with keys 'pehe', 'ate_error', 'ate_pred', 'ate_true', 'relative_ate_error'.
    """
    tau_true = Y_true_dict[treatment_to] - Y_true_dict[treatment_from]
    tau_pred = tlearner.predict(X, treatment_from, treatment_to)
    pehe = calculate_pehe(tau_pred, tau_true)
    ate_error = calculate_ate_error(tau_pred, tau_true)
    ate_pred = np.mean(tau_pred)
    ate_true = np.mean(tau_true)
    
    return {
        'pehe': pehe,
        'ate_error': ate_error,
        'ate_pred': ate_pred,
        'ate_true': ate_true,
        'relative_ate_error': ate_error / abs(ate_true) if ate_true != 0 else float('inf')
    }
 

def evaluate_clinical_threshold_policy(model, X, Y_true_dict, current_treatments, threshold):
    """
    @brief Evaluate a policy that only switches treatment when improvement exceeds a threshold.

    @param model                Fitted model with a predict_outcome() method.
    @param X                   Feature matrix of shape (n_samples, n_features).
    @param Y_true_dict         Dict mapping treatment index to true outcomes array.
    @param current_treatments  Array of each patient's current treatment.
    @param threshold           Minimum predicted improvement required to switch.
    @return Dict with policy evaluation metrics (policy_value, oracle_value, accuracy, etc.).
    """
    treatment_values = sorted(Y_true_dict.keys())
    n_samples = len(X)
    current_treatments = np.array(current_treatments)
    
    # Predict outcomes for all treatments
    pred_outcomes = {t: model.predict_outcome(X, t) for t in treatment_values}
    
    # For each patient, find the best predicted treatment
    pred_matrix = np.column_stack([pred_outcomes[t] for t in treatment_values])
    best_pred_idx = np.argmax(pred_matrix, axis=1)
    best_pred_treatments = np.array([treatment_values[i] for i in best_pred_idx])
    best_pred_outcomes = np.max(pred_matrix, axis=1)
    
    # Get predicted outcome under current treatment
    current_pred_outcomes = np.array([pred_outcomes[current_treatments[i]][i] for i in range(n_samples)])
    
    # Calculate predicted improvement
    predicted_improvement = best_pred_outcomes - current_pred_outcomes
    
    # Apply threshold: only switch if improvement > threshold
    recommended_treatments = np.where(
        predicted_improvement >= threshold,
        best_pred_treatments,  # Switch to best
        current_treatments      # Keep current
    )
    
    # Calculate true outcomes under recommended treatments
    policy_outcomes = np.array([Y_true_dict[recommended_treatments[i]][i] for i in range(n_samples)])
    
    # Calculate true outcomes under current treatments (no change)
    current_outcomes = np.array([Y_true_dict[current_treatments[i]][i] for i in range(n_samples)])
    
    # Oracle: always pick true best
    true_matrix = np.column_stack([Y_true_dict[t] for t in treatment_values])
    oracle_outcomes = np.max(true_matrix, axis=1)
    oracle_treatments = np.array([treatment_values[i] for i in np.argmax(true_matrix, axis=1)])
    
    # Count switches
    n_switched = np.sum(recommended_treatments != current_treatments)
    n_correct = np.sum(recommended_treatments == oracle_treatments)
    
    return {
        'policy_value': np.mean(policy_outcomes),
        'current_value': np.mean(current_outcomes),
        'oracle_value': np.mean(oracle_outcomes),
        'improvement': np.mean(policy_outcomes) - np.mean(current_outcomes),
        'n_switched': n_switched,
        'pct_switched': n_switched / n_samples * 100,
        'accuracy': n_correct / n_samples * 100,
        'recommended_treatments': recommended_treatments,
        'predicted_improvements': predicted_improvement
    }

