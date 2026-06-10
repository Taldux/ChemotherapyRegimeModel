"""
Unit tests for src/models/metrics.py.

Covers: calculate_pehe, calculate_ate_error,
        evaluate_semi_synthetic, evaluate_clinical_threshold_policy.
"""

import numpy as np
import pytest

from src.models.metrics import (
    calculate_ate_error,
    calculate_pehe,
    evaluate_clinical_threshold_policy,
    evaluate_semi_synthetic,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ConstantPredictor:
    """Mock model: predict() returns a constant ITE; predict_outcome() returns arm mean."""

    def __init__(self, ite_value: float, arm_means: dict):
        self._ite = ite_value
        self._means = arm_means

    def predict(self, X, treatment_from, treatment_to):
        return np.full(len(X), self._ite)

    def predict_outcome(self, X, treatment):
        return np.full(len(X), self._means[treatment])


# ---------------------------------------------------------------------------
# calculate_pehe
# ---------------------------------------------------------------------------

def test_pehe_perfect_prediction():
    arr = np.array([1.0, 2.0, 3.0])
    assert calculate_pehe(arr, arr) == pytest.approx(0.0)


def test_pehe_known_value():
    # pred - true = [1, 1]  →  sqrt(mean(1)) = 1.0
    pred = np.array([2.0, 4.0])
    true = np.array([1.0, 3.0])
    assert calculate_pehe(pred, true) == pytest.approx(1.0)


def test_pehe_accepts_lists():
    assert calculate_pehe([0.0, 0.0], [0.0, 0.0]) == pytest.approx(0.0)


def test_pehe_nonnegative():
    rng = np.random.default_rng(0)
    pred = rng.standard_normal(50)
    true = rng.standard_normal(50)
    assert calculate_pehe(pred, true) >= 0.0


def test_pehe_symmetric():
    pred = np.array([1.0, 2.0, 3.0])
    true = np.array([4.0, 5.0, 6.0])
    assert calculate_pehe(pred, true) == pytest.approx(calculate_pehe(true, pred))


# ---------------------------------------------------------------------------
# calculate_ate_error
# ---------------------------------------------------------------------------

def test_ate_error_zero():
    arr = np.array([1.0, -1.0, 2.0])
    assert calculate_ate_error(arr, arr) == pytest.approx(0.0)


def test_ate_error_known():
    # mean(pred) = 3, mean(true) = 1  →  |3 - 1| = 2
    pred = np.array([3.0, 3.0])
    true = np.array([1.0, 1.0])
    assert calculate_ate_error(pred, true) == pytest.approx(2.0)


def test_ate_error_nonnegative():
    rng = np.random.default_rng(1)
    pred = rng.standard_normal(40)
    true = rng.standard_normal(40)
    assert calculate_ate_error(pred, true) >= 0.0


def test_ate_error_symmetric():
    pred = np.array([5.0, 5.0])
    true = np.array([2.0, 2.0])
    assert calculate_ate_error(pred, true) == pytest.approx(
        calculate_ate_error(true, pred)
    )


# ---------------------------------------------------------------------------
# evaluate_semi_synthetic
# ---------------------------------------------------------------------------

def test_evaluate_semi_synthetic_returns_expected_keys(rng):
    n = 20
    X = rng.standard_normal((n, 3))
    y0 = rng.uniform(0, 50, n)
    y1 = y0 + 10.0
    Y_true = {0: y0, 1: y1}

    model = _ConstantPredictor(ite_value=10.0, arm_means={0: 0.0, 1: 10.0})
    result = evaluate_semi_synthetic(model, X, Y_true, 0, 1)

    assert set(result.keys()) == {
        "pehe", "ate_error", "ate_pred", "ate_true", "relative_ate_error"
    }


def test_evaluate_semi_synthetic_perfect_model(rng):
    """A model that always predicts the true ITE should have pehe ≈ 0."""
    n = 30
    X = rng.standard_normal((n, 4))
    y0 = np.zeros(n)
    y1 = np.full(n, 7.0)   # true ITE = 7 for every sample
    Y_true = {0: y0, 1: y1}

    model = _ConstantPredictor(ite_value=7.0, arm_means={0: 0.0, 1: 7.0})
    result = evaluate_semi_synthetic(model, X, Y_true, 0, 1)

    assert result["pehe"] == pytest.approx(0.0, abs=1e-9)
    assert result["ate_error"] == pytest.approx(0.0, abs=1e-9)
    assert result["ate_pred"] == pytest.approx(7.0)
    assert result["ate_true"] == pytest.approx(7.0)


def test_evaluate_semi_synthetic_relative_ate_error_zero_true(rng):
    """When true ATE is 0, relative_ate_error should be inf."""
    n = 10
    X = rng.standard_normal((n, 3))
    y_same = np.ones(n) * 5.0
    Y_true = {0: y_same, 1: y_same}   # ATE_true = 0

    model = _ConstantPredictor(ite_value=1.0, arm_means={0: 5.0, 1: 6.0})
    result = evaluate_semi_synthetic(model, X, Y_true, 0, 1)

    assert result["relative_ate_error"] == float("inf")


# ---------------------------------------------------------------------------
# evaluate_clinical_threshold_policy
# ---------------------------------------------------------------------------

def test_clinical_policy_no_switch_at_infinite_threshold(rng):
    """Threshold=inf means nobody switches, so policy_value == current_value."""
    n = 20
    X = rng.standard_normal((n, 4))
    y0 = np.full(n, 40.0)
    y1 = np.full(n, 60.0)
    Y_true = {0: y0, 1: y1}
    current = np.zeros(n, dtype=int)  # everyone on arm 0

    model = _ConstantPredictor(ite_value=20.0, arm_means={0: 40.0, 1: 60.0})
    result = evaluate_clinical_threshold_policy(model, X, Y_true, current, threshold=float("inf"))

    assert result["n_switched"] == 0
    assert result["policy_value"] == pytest.approx(result["current_value"])


def test_clinical_policy_all_switch_at_zero_threshold(rng):
    """
    Model perfectly predicts arm 1 is better by 20 units.
    With threshold=0 everyone should switch to arm 1.
    """
    n = 20
    X = rng.standard_normal((n, 4))
    y0 = np.full(n, 40.0)
    y1 = np.full(n, 60.0)
    Y_true = {0: y0, 1: y1}
    current = np.zeros(n, dtype=int)

    model = _ConstantPredictor(ite_value=20.0, arm_means={0: 40.0, 1: 60.0})
    result = evaluate_clinical_threshold_policy(model, X, Y_true, current, threshold=0)

    assert result["n_switched"] == n
    assert result["policy_value"] == pytest.approx(60.0)


def test_clinical_policy_result_keys(rng):
    n = 10
    X = rng.standard_normal((n, 3))
    Y_true = {0: np.zeros(n), 1: np.ones(n) * 5.0}
    current = np.zeros(n, dtype=int)
    model = _ConstantPredictor(ite_value=5.0, arm_means={0: 0.0, 1: 5.0})

    result = evaluate_clinical_threshold_policy(model, X, Y_true, current, threshold=0)

    expected_keys = {
        "policy_value", "current_value", "oracle_value",
        "improvement", "n_switched", "pct_switched",
        "accuracy", "recommended_treatments", "predicted_improvements",
    }
    assert set(result.keys()) == expected_keys


def test_clinical_policy_oracle_is_upper_bound(rng):
    """Policy value must never exceed oracle value."""
    n = 30
    X = rng.standard_normal((n, 5))
    y0 = rng.uniform(0, 50, n)
    y1 = rng.uniform(0, 50, n)
    Y_true = {0: y0, 1: y1}
    current = np.zeros(n, dtype=int)
    model = _ConstantPredictor(ite_value=3.0, arm_means={0: 20.0, 1: 23.0})

    result = evaluate_clinical_threshold_policy(model, X, Y_true, current, threshold=0)

    assert result["policy_value"] <= result["oracle_value"] + 1e-9
