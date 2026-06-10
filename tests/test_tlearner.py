"""
Unit tests for src/models/tlearner.py.

Covers: fit, predict_outcome, predict, error handling.
"""

import numpy as np
import pytest

from src.models.tlearner import TLearner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(rng, n_per_arm=30, n_features=5, offsets=(0.0, 5.0, 10.0)):
    """Return (X, t, y) with a known per-arm linear outcome."""
    n = n_per_arm * len(offsets)
    X = rng.standard_normal((n, n_features))
    t = np.repeat(range(len(offsets)), n_per_arm)
    offsets = np.array(offsets)
    y = X[:, 0] + offsets[t] + rng.standard_normal(n) * 0.01
    return X, t, y


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def test_fit_stores_treatment_values(rng, linear_base):
    X, t, y = _make_data(rng)
    model = TLearner(base_model=linear_base).fit(X, t, y)
    assert set(model.treatment_values) == {0, 1, 2}


def test_fit_creates_one_model_per_arm(rng, linear_base):
    X, t, y = _make_data(rng)
    model = TLearner(base_model=linear_base).fit(X, t, y)
    assert set(model.models.keys()) == {0, 1, 2}


def test_fit_only_two_arms_rejects_third_on_predict(rng, linear_base):
    """A model trained on arms 0 and 1 must raise ValueError for arm 2."""
    X, t, y = _make_data(rng, offsets=(0.0, 5.0))   # only arms 0 and 1
    model = TLearner(base_model=linear_base).fit(X, t, y)
    assert set(model.treatment_values) == {0, 1}
    X_test = rng.standard_normal((5, X.shape[1]))
    with pytest.raises(ValueError, match="not seen during training"):
        model.predict_outcome(X_test, treatment=2)


# ---------------------------------------------------------------------------
# predict_outcome
# ---------------------------------------------------------------------------

def test_predict_outcome_shape(rng, linear_base):
    X, t, y = _make_data(rng)
    model = TLearner(base_model=linear_base).fit(X, t, y)
    X_test = rng.standard_normal((15, X.shape[1]))
    preds = model.predict_outcome(X_test, treatment=0)
    assert preds.shape == (15,)


def test_predict_outcome_unseen_treatment_raises(rng, linear_base):
    X, t, y = _make_data(rng)
    model = TLearner(base_model=linear_base).fit(X, t, y)
    X_test = rng.standard_normal((5, X.shape[1]))
    with pytest.raises(ValueError, match="not seen during training"):
        model.predict_outcome(X_test, treatment=99)


# ---------------------------------------------------------------------------
# predict (ITE)
# ---------------------------------------------------------------------------

def test_predict_shape(rng, linear_base):
    X, t, y = _make_data(rng)
    model = TLearner(base_model=linear_base).fit(X, t, y)
    X_test = rng.standard_normal((15, X.shape[1]))
    ite = model.predict(X_test, treatment_from=0, treatment_to=1)
    assert ite.shape == (15,)


def test_predict_direction(rng, linear_base):
    """
    With arm offsets 0, 5, 10 and very low noise,
    predict(0 → 2) should be positive and around 10.
    """
    X, t, y = _make_data(rng, n_per_arm=100, offsets=(0.0, 5.0, 10.0))
    model = TLearner(base_model=linear_base).fit(X, t, y)
    X_test = rng.standard_normal((50, X.shape[1]))
    ite = model.predict(X_test, treatment_from=0, treatment_to=2)
    assert np.mean(ite) == pytest.approx(10.0, abs=0.5)


def test_predict_antisymmetric(rng, linear_base):
    """predict(from=0, to=1) == -predict(from=1, to=0)."""
    X, t, y = _make_data(rng, n_per_arm=50)
    model = TLearner(base_model=linear_base).fit(X, t, y)
    X_test = rng.standard_normal((20, X.shape[1]))
    assert model.predict(X_test, 0, 1) == pytest.approx(
        -model.predict(X_test, 1, 0), abs=1e-9
    )


def test_predict_self_is_zero(rng, linear_base):
    """ITE from arm k to itself must be zero."""
    X, t, y = _make_data(rng)
    model = TLearner(base_model=linear_base).fit(X, t, y)
    X_test = rng.standard_normal((10, X.shape[1]))
    ite = model.predict(X_test, treatment_from=1, treatment_to=1)
    np.testing.assert_allclose(ite, 0.0, atol=1e-9)


def test_predict_unseen_from_treatment_raises(rng, linear_base):
    X, t, y = _make_data(rng)
    model = TLearner(base_model=linear_base).fit(X, t, y)
    X_test = rng.standard_normal((5, X.shape[1]))
    with pytest.raises(ValueError, match="not seen during training"):
        model.predict(X_test, treatment_from=99, treatment_to=0)


def test_predict_unseen_to_treatment_raises(rng, linear_base):
    X, t, y = _make_data(rng)
    model = TLearner(base_model=linear_base).fit(X, t, y)
    X_test = rng.standard_normal((5, X.shape[1]))
    with pytest.raises(ValueError, match="not seen during training"):
        model.predict(X_test, treatment_from=0, treatment_to=99)
