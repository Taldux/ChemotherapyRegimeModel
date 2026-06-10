"""
Unit tests for src/models/panel_utils.py.

Covers: balance_panel, LOCF padding, truncation, X=None, W provided.
"""

import numpy as np
import pytest

from src.models.panel_utils import balance_panel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_panels(sizes, n_features=3, seed=0):
    """
    Build flat (Y, T, X, groups) arrays from a list of per-group sizes.

    Groups are labelled 0, 1, 2, … in order. Values are simple integers
    so assertions are easy to reason about.
    """
    rng = np.random.default_rng(seed)
    groups = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    n = len(groups)
    Y = rng.standard_normal(n)
    T = rng.integers(0, 3, size=n)
    X = rng.standard_normal((n, n_features))
    return Y, T, X, groups


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------

def test_equal_groups_output_shape():
    sizes = [3, 3, 3]
    Y, T, X, groups = _make_panels(sizes)
    Y_b, T_b, X_b, G_b, W_b = balance_panel(Y, T, X, groups, max_periods=3)

    n_groups = len(sizes)
    assert Y_b.shape == (n_groups * 3,)
    assert T_b.shape == (n_groups * 3,)
    assert X_b.shape == (n_groups * 3, X.shape[1])
    assert G_b.shape == (n_groups * 3,)
    assert W_b is None


def test_w_none_returns_none():
    Y, T, X, groups = _make_panels([4, 4])
    _, _, _, _, W_b = balance_panel(Y, T, X, groups, max_periods=4)
    assert W_b is None


def test_w_provided_shape():
    Y, T, X, groups = _make_panels([3, 3])
    W = np.ones((len(Y), 2))
    _, _, _, _, W_b = balance_panel(Y, T, X, groups, W=W, max_periods=3)
    assert W_b.shape == (2 * 3, 2)


def test_x_none_output():
    Y, T, _, groups = _make_panels([4, 4])
    Y_b, T_b, X_b, G_b, W_b = balance_panel(Y, T, None, groups, max_periods=4)
    assert X_b is None


# ---------------------------------------------------------------------------
# LOCF padding (short groups)
# ---------------------------------------------------------------------------

def test_locf_padding_values():
    """
    A group of size 1 padded to max_periods=3 should repeat the single row
    in positions 1 and 2.
    """
    # Group 0: one row  |  Group 1: three rows
    Y = np.array([99.0,  1.0, 2.0, 3.0])
    T = np.array([7,     0,   1,   2])
    X = np.array([[9.0], [1.0], [2.0], [3.0]])
    groups = np.array([0, 1, 1, 1])

    Y_b, T_b, X_b, G_b, _ = balance_panel(Y, T, X, groups, max_periods=3)

    # Group 0 occupies indices 0-2 → all three should be the LOCF value (99)
    np.testing.assert_array_equal(Y_b[:3], [99.0, 99.0, 99.0])
    np.testing.assert_array_equal(T_b[:3], [7, 7, 7])
    np.testing.assert_array_equal(X_b[:3, 0], [9.0, 9.0, 9.0])


def test_locf_group_ids_correct():
    """G_bal should repeat the group id for every padded slot."""
    Y = np.array([1.0, 2.0, 3.0, 4.0])
    T = np.array([0, 1, 0, 1])
    X = np.ones((4, 2))
    groups = np.array([0, 1, 1, 1])

    _, _, _, G_b, _ = balance_panel(Y, T, X, groups, max_periods=3)

    assert list(G_b[:3]) == [0, 0, 0]   # padded group 0
    assert list(G_b[3:]) == [1, 1, 1]   # untouched group 1


# ---------------------------------------------------------------------------
# Truncation (long groups)
# ---------------------------------------------------------------------------

def test_truncation_to_max_periods():
    """A group with 5 rows truncated to max_periods=2 should keep first 2."""
    Y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    T = np.array([0, 1, 2, 0, 1])
    X = np.arange(5).reshape(5, 1).astype(float)
    groups = np.zeros(5, dtype=int)

    Y_b, T_b, X_b, G_b, _ = balance_panel(Y, T, X, groups, max_periods=2)

    assert Y_b.shape == (2,)
    np.testing.assert_array_equal(Y_b, [10.0, 20.0])


# ---------------------------------------------------------------------------
# max_periods defaults and minimum floor
# ---------------------------------------------------------------------------

def test_max_periods_defaults_to_median():
    """
    With three groups of sizes [2, 3, 4], median=3.
    Passing max_periods=None should use ceil(median)=3.
    """
    sizes = [2, 3, 4]
    Y, T, X, groups = _make_panels(sizes)
    Y_b, _, _, _, _ = balance_panel(Y, T, X, groups)  # max_periods=None

    n_groups = len(sizes)
    assert Y_b.shape[0] == n_groups * 3


def test_max_periods_clamped_to_minimum_two():
    """
    Even if every group has only 1 row, max_periods must be at least 2.
    """
    Y = np.array([1.0, 2.0])
    T = np.array([0, 1])
    X = np.ones((2, 1))
    groups = np.array([0, 1])

    Y_b, _, _, _, _ = balance_panel(Y, T, X, groups, max_periods=1)

    # Two groups × max(1, 2) = 4 rows
    assert Y_b.shape[0] == 4
