"""Tests for energy functions."""
import pytest
import os
import pandas as pd
from pandas.testing import assert_series_equal
from pvanalytics.quality import energy

script_directory = os.path.dirname(__file__)
energy_filepath = os.path.join(script_directory,
                               "../../data/system_10004_ac_energy.csv")
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


def test_convert_cumulative_with_simple_diff(cumulative_series,
                                             simple_diff_energy_series):
    """
    Tests convert_cumulative_energy for cumulative series.
    Test returns the corrected differenced series via simple differencing.
    """
    simple_diff_result = energy.convert_cumulative_energy(
        energy_series=cumulative_series, system_self_consumption=0.0)
    assert_series_equal(simple_diff_result, simple_diff_energy_series)


def test_convert_cumulative_with_avg_diff(cumulative_series,
                                          avg_diff_energy_series):
    """
    Tests convert_cumulative_energy for cumulative series.
    Test returns the corrected differenced series via avgerage differencing.
    """
    simple_diff_result = energy.convert_cumulative_energy(
        energy_series=avg_diff_energy_series, system_self_consumption=0.0)
    assert_series_equal(simple_diff_result, avg_diff_energy_series)


def test_convert_noncumulative(noncumulative_series):
    """
    Tests convert_cumulative_energy for non-cumulative series.
    Test returns the original non-cumulative energy series.
    """
    energy_result = energy.convert_cumulative_energy(
        energy_series=noncumulative_series, system_self_consumption=0.0)
    assert_series_equal(energy_result, noncumulative_series)


def test_check_zero_length(zero_length_series):
    """
    Tests convert_cumulative_energy for zero length series.
    Test returns the original zero length energy series.
    """
    energy_result = energy.convert_cumulative_energy(
        energy_series=zero_length_series, system_self_consumption=0.0)
    assert_series_equal(energy_result, zero_length_series)
