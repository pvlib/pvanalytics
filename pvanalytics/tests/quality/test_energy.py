"""Tests for energy functions."""
import pytest
import os
import pandas as pd
from pandas.testing import assert_series_equal
from pvanalytics.quality import energy

script_directory = os.path.dirname(__file__)
energy_filepath = os.path.join(script_directory,
                               "../../data/10004_one_week.csv")
energy_df = pd.read_csv(energy_filepath)


@pytest.fixture
def cumulative_series():
    """
    A pandas energy time series with cumulative data.
    """
    data = energy_df["ac_energy_inv_16425"]
    return pd.Series(data=data)


@pytest.fixture
def noncumulative_series():
    """
    A pandas energy time series with noncumulative data.
    """
    # Perform .diff() to turn cumulative data into noncumulative
    diff_data = energy_df["ac_energy_inv_16425"].diff().dropna()
    return pd.Series(data=diff_data)


@pytest.fixture
def zero_length_series():
    """
    A pandas energy time series that has zero length.
    """
    data = []
    return pd.Series(data=data, dtype="float64")


@pytest.fixture
def simple_diff_energy_series():
    """
    The differenced pandas energy series using the simple .diff() function.
    """
    diff_data = energy_df["ac_energy_inv_16425"].diff()
    return pd.Series(data=diff_data)


@pytest.fixture
def avg_diff_energy_series():
    """
    The differenced pandas energy series using averaged difference method.
    """
    diff_data = energy_df["ac_energy_inv_16425"].diff()
    avg_diff_series = 0.5 * (diff_data.shift(-1) + diff_data)
    return pd.Series(data=avg_diff_series)


def test_cumulative_energy_simple_diff_check_true(cumulative_series):
    """
    Tests if cumulative_energy_simple_diff_check for cumulative series is True.
    """
    assert energy.cumulative_energy_simple_diff_check(
        energy_series=cumulative_series, system_self_consumption=0.0) is True


def test_noncumulative_energy_simple_diff_check_false(noncumulative_series):
    """
    Tests if cumulative_energy_simple_diff_check for noncumulative series
    is False.
    """
    assert energy.cumulative_energy_simple_diff_check(
        energy_series=noncumulative_series,
        system_self_consumption=0.0) is False


def test_zero_length_energy_simple_diff_check_false(zero_length_series):
    """
    Tests if cumulative_energy_simple_diff_check for zero length series is
    False.
    """
    assert energy.cumulative_energy_simple_diff_check(
        energy_series=zero_length_series) is False


def test_cumulative_energy_avg_diff_check_true(cumulative_series):
    """
    Tests if cumulative_energy_avg_diff_check for cumulative series is True.
    """
    assert energy.cumulative_energy_avg_diff_check(
        energy_series=cumulative_series, system_self_consumption=0.0) is True


def test_noncumulative_energy_avg_diff_check_false(noncumulative_series):
    """
    Tests if cumulative_energy_avg_diff_check for noncumulative series
    is False.
    """
    assert energy.cumulative_energy_avg_diff_check(
        energy_series=noncumulative_series,
        system_self_consumption=0.0) is False


def test_zero_length_energy_avg_diff_check_false(zero_length_series):
    """
    Tests if cumulative_energy_avg_diff_check for zero length series is
    False.
    """
    assert energy.cumulative_energy_avg_diff_check(
        energy_series=zero_length_series) is False


def test_check_cumulative(cumulative_series, simple_diff_energy_series,
                          avg_diff_energy_series):
    """
    Tests check_cumulative_energy for cumulative series.
    Test returns the adjusted difference series, average difference series,
    and True.
    """
    simple_diff_result, avg_diff_result, is_cumulative = \
        energy.check_cumulative_energy(
            energy_series=cumulative_series, system_self_consumption=0.0)
    assert_series_equal(simple_diff_result, simple_diff_energy_series)
    assert_series_equal(avg_diff_result, avg_diff_energy_series)
    assert is_cumulative is True


def test_check_noncumulative(noncumulative_series):
    """
    Tests check_cumulative_energy for noncumulative series.
    Test returns the original noncumulative energy series, original
    noncumulative energy series, and False.
    """
    simple_diff_result, avg_diff_result, is_cumulative = \
        energy.check_cumulative_energy(
            energy_series=noncumulative_series, system_self_consumption=0.0)
    assert_series_equal(simple_diff_result, noncumulative_series)
    assert_series_equal(avg_diff_result, noncumulative_series)
    assert is_cumulative is False


def test_check_zero_length(zero_length_series):
    """
    Tests check_cumulative_energy for zero length series.
    Test returns the original zero length energy series, original
    zero length energy series, and False.
    """
    simple_diff_result, avg_diff_result, is_cumulative = \
        energy.check_cumulative_energy(
            energy_series=zero_length_series, system_self_consumption=0.0)
    assert_series_equal(simple_diff_result, zero_length_series)
    assert_series_equal(avg_diff_result, zero_length_series)
    assert is_cumulative is False
