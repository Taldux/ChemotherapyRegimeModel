"""
Unit tests for src/models/slearner.py.

Covers: fit, predict_outcome, predict.
"""

import numpy as np
import pytest

from src.models.slearner import SLearner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(rng, n_per_arm=40, n_features=5, offsets=(0.0, 5.0, 10.0)):
    """Return (X, t, y) with a known per-arm linear outcome."""
    n = n_per_arm * len(offsets)
    X = rng.standard_normal((n, n_features))
    t = np.repeat(range(len(offsets)), n_per_arm)
    offsets_arr = np.array(offsets)
    y = X[:, 0] + offsets_arr[t] + rng.standard_normal(n) * 0.01
    return X, t, y.ravel()


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def test_fit_stores_treatment_values(rng, linear_base):
    X, t, y = _make_data(rng)
    model = SLearner(base_model=linear_base).fit(X, t, y)
    assert set(model.treatment_values) == {0, 1, 2}


def test_fit_trains_single_model(rng, linear_base):
    X, t, y = _make_data(rng)
    model = SLearner(base_model=linear_base).fit(X, t, y)
    assert model.model is not None


def test_fit_returns_self(rng, linear_base):
    X, t, y = _make_data(rng)
    instance = SLearner(base_model=linear_base)
    assert instance.fit(X, t, y) is instance


# ---------------------------------------------------------------------------
# predict_outcome
# ---------------------------------------------------------------------------

def test_predict_outcome_shape(rng, linear_base):
    X, t, y = _make_data(rng)
    model = SLearner(base_model=linear_base).fit(X, t, y)
    X_test = rng.standard_normal((15, X.shape[1]))
    preds = model.predict_outcome(X_test, treatment=0)
    assert preds.shape == (15,)


def test_predict_outcome_higher_for_better_arm(rng, linear_base):
    """
    With large arm offsets, arm 2 predictions should exceed arm 0 predictions
    on average.
    """
    X, t, y = _make_data(rng, n_per_arm=100, offsets=(0.0, 5.0, 10.0))
    model = SLearner(base_model=linear_base).fit(X, t, y)
    X_test = rng.standard_normal((50, X.shape[1]))
    assert np.mean(model.predict_outcome(X_test, 2)) > np.mean(
        model.predict_outcome(X_test, 0)
    )


# ---------------------------------------------------------------------------
# predict (ITE)
# ---------------------------------------------------------------------------

def test_predict_shape(rng, linear_base):
    X, t, y = _make_data(rng)
    model = SLearner(base_model=linear_base).fit(X, t, y)
    X_test = rng.standard_normal((15, X.shape[1]))
    ite = model.predict(X_test, from_treatment=0, to_treatment=1)
    assert ite.shape == (15,)


def test_predict_direction(rng, linear_base):
    """
    With offsets 0, 5, 10 and low noise, mean ITE (0→2) should be ≈ 10.
    """
    X, t, y = _make_data(rng, n_per_arm=120, offsets=(0.0, 5.0, 10.0))
    model = SLearner(base_model=linear_base).fit(X, t, y)
    X_test = rng.standard_normal((60, X.shape[1]))
    ite = model.predict(X_test, from_treatment=0, to_treatment=2)
    assert np.mean(ite) == pytest.approx(10.0, abs=0.5)


def test_predict_antisymmetric(rng, linear_base):
    """predict(from=0, to=1) == -predict(from=1, to=0)."""
    X, t, y = _make_data(rng)
    model = SLearner(base_model=linear_base).fit(X, t, y)
    X_test = rng.standard_normal((20, X.shape[1]))
    np.testing.assert_allclose(
        model.predict(X_test, 0, 1),
        -model.predict(X_test, 1, 0),
        atol=1e-9,
    )


def test_predict_self_is_zero(rng, linear_base):
    """ITE from arm k to itself must be zero."""
    X, t, y = _make_data(rng)
    model = SLearner(base_model=linear_base).fit(X, t, y)
    X_test = rng.standard_normal((10, X.shape[1]))
    ite = model.predict(X_test, from_treatment=1, to_treatment=1)
    np.testing.assert_allclose(ite, 0.0, atol=1e-9)
