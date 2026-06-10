"""
Shared pytest fixtures for the ChemotherapyRegimeModel test suite.
"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression


RNG_SEED = 42


@pytest.fixture
def rng():
    """Reproducible NumPy random generator."""
    return np.random.default_rng(RNG_SEED)


@pytest.fixture
def three_arm_data(rng):
    """
    Small three-treatment dataset for model tests.

    Returns (X, t, y, Y_true_dict) where treatment 2 always gives the
    highest outcome so the oracle recommendation is always treatment 2.
    """
    n = 60  # 20 samples per arm
    n_features = 5

    X = rng.standard_normal((n, n_features))
    t = np.repeat([0, 1, 2], n // 3)

    # Linear outcome with known treatment offsets: arm 0 → +0, arm 1 → +5, arm 2 → +10
    offsets = np.array([0.0, 5.0, 10.0])
    y = X[:, 0] + offsets[t] + rng.standard_normal(n) * 0.1

    # Ground-truth potential outcomes for each arm (evaluated on all n rows)
    Y_true_dict = {
        0: X[:, 0] + offsets[0],
        1: X[:, 0] + offsets[1],
        2: X[:, 0] + offsets[2],
    }

    return X, t, y, Y_true_dict


@pytest.fixture
def linear_base():
    """Fast sklearn base model for learner tests."""
    return LinearRegression()
