"""Tests for energy functions."""
import pytest
import pandas as pd
from pandas.testing import assert_series_equal
from pvanalytics.quality import energy


@pytest.fixture
def cumulative_series():
    """
    A series that is cumulative data.
    """
    data = [0, 10, 25, 30, 45, 60]
    return pd.Series(data=data)


@pytest.fixture
def noncumulative_series():
    """A series that is noncumulative data.
    """
    data = [0, 10, 5, 15, 0, 20]
    return pd.Series(data=data)


@pytest.fixture
def zero_length_series():
    """
    A series that has zero length.
    """
    data = []
    return pd.Series(data=data, dtype="float64")


def test_is_cumulative_energy_true(cumulative_series):
    """
    Tests if is_cumulative_energy for cumulative series is True.
    """
    assert energy.is_cumulative_energy(cumulative_series) is True


def test_is_noncumulative_energy_false(noncumulative_series):
    """
    Tests if is_cumulative_energy for noncumulative series is False.
    """
    assert energy.is_cumulative_energy(noncumulative_series) is False


def test_is_zero_length_cumulative_energy_false(zero_length_series):
    """
    Tests if is_cumulative_energy for zero length series is False.
    """
    assert energy.is_cumulative_energy(zero_length_series) is False


def test_check_cumulative(cumulative_series):
    """
    Tests check_cumulative_energy for cumulative series.
    Test returns the adjusted series and True.
    """
    result_series, is_cumulative = energy.check_cumulative_energy(
        cumulative_series)
    adjusted_series = pd.Series([None, 10, 15, 5, 15, 15])
    assert_series_equal(result_series, adjusted_series)
    assert is_cumulative is True


def test_check_noncumulative(noncumulative_series):
    """
    Tests check_cumulative_energy for noncumulative series.
    Test returns the nonadjusted series (the same as noncumulative series)
    and False.
    """
    result_series, is_cumulative = energy.check_cumulative_energy(
        noncumulative_series)
    assert_series_equal(result_series, noncumulative_series)
    assert is_cumulative is False


def test_check_zero_length(zero_length_series):
    """
    Tests check_cumulative_energy for zero length series.
    Test returns the nonadjusted series (the same as noncumulative series)
    and False.
    """
    result_series, is_cumulative = energy.check_cumulative_energy(
        zero_length_series)
    assert_series_equal(result_series, zero_length_series)
    assert is_cumulative is False
