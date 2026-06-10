"""
Unit tests for src/preprocess/preprocess.py.

Covers: normalize_regime, normalize_regime_column.
"""

import pandas as pd
import pytest

from src.preprocess.preprocess import normalize_regime, normalize_regime_column


# ---------------------------------------------------------------------------
# normalize_regime
# ---------------------------------------------------------------------------

def test_normalize_regime_already_sorted():
    assert normalize_regime("Cisplatin/Navelbine") == "Cisplatin/Navelbine"


def test_normalize_regime_sorts_components():
    # 'Navelbine' < 'Cisplatin' alphabetically? No: C < N, so Cisplatin comes first
    assert normalize_regime("Navelbine/Cisplatin") == "Cisplatin/Navelbine"


def test_normalize_regime_three_components():
    result = normalize_regime("C/B/A")
    assert result == "A/B/C"


def test_normalize_regime_single_component():
    assert normalize_regime("Navelbine") == "Navelbine"


def test_normalize_regime_strips_whitespace():
    # Components with leading/trailing spaces should be stripped before sorting
    assert normalize_regime("B / A") == "A/B"


def test_normalize_regime_case_sensitive():
    # Sorting is case-sensitive; uppercase letters sort before lowercase in ASCII
    result = normalize_regime("b/A")
    assert result == "A/b"


def test_normalize_regime_idempotent():
    regime = "Alpha/Beta/Gamma"
    assert normalize_regime(normalize_regime(regime)) == normalize_regime(regime)


# ---------------------------------------------------------------------------
# normalize_regime_column
# ---------------------------------------------------------------------------

def test_normalize_regime_column_modifies_inplace():
    df = pd.DataFrame({"Chemotherapieregime": ["B/A", "C/A", "A/B"]})
    result = normalize_regime_column(df)
    assert result is df  # same object returned


def test_normalize_regime_column_values_sorted():
    df = pd.DataFrame({"Chemotherapieregime": ["Navelbine/Cisplatin", "Cisplatin/Navelbine"]})
    normalize_regime_column(df)
    assert (df["Chemotherapieregime"] == "Cisplatin/Navelbine").all()


def test_normalize_regime_column_reduces_unique_count():
    # 'B/A' and 'A/B' are duplicates after normalization → 1 unique instead of 2
    df = pd.DataFrame({"Chemotherapieregime": ["B/A", "A/B", "C/A", "A/C"]})
    normalize_regime_column(df)
    assert df["Chemotherapieregime"].nunique() == 2


def test_normalize_regime_column_handles_nan():
    """NaN values should not raise; they are left empty (falsy branch)."""
    df = pd.DataFrame({"Chemotherapieregime": ["B/A", None]})
    normalize_regime_column(df)
    assert df["Chemotherapieregime"].iloc[0] == "A/B"
    # Second row should be empty string (fillna("") result)
    assert df["Chemotherapieregime"].iloc[1] == ""


def test_normalize_regime_column_custom_col_name():
    df = pd.DataFrame({"regime": ["B/A", "A/C"]})
    normalize_regime_column(df, col="regime")
    assert df["regime"].tolist() == ["A/B", "A/C"]
