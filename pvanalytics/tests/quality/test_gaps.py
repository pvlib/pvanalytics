"""Tests for gaps quality control functions."""
import pytest
import pandas as pd
from pandas.util.testing import assert_series_equal
from pvanalytics.quality import gaps


@pytest.fixture
def stale_data():
    """A series that contains stuck values."""
    data = [1.0, 1.001, 1.001, 1.001, 1.001, 1.001001, 1.001, 1.001, 1.2, 1.3]
    return pd.Series(data=data)


@pytest.fixture
def data_with_negatives():
    """A series that contains stuck values, interpolation, and negatives."""
    data = [0.0, 0.0, 0.0, -0.0, 0.00001, 0.000010001, -0.00000001]
    return pd.Series(data=data)


def test_detect_stale_values(stale_data):
    """detect_stale_values properly identifies stuck values."""
    res1 = gaps.detect_stale_values(stale_data)
    res2 = gaps.detect_stale_values(stale_data, rtol=1e-8, window=2)
    res3 = gaps.detect_stale_values(stale_data, window=7)
    res4 = gaps.detect_stale_values(stale_data, window=8)
    res5 = gaps.detect_stale_values(stale_data, rtol=1e-8, window=4)
    res6 = gaps.detect_stale_values(stale_data[1:])
    res7 = gaps.detect_stale_values(stale_data[1:8])
    assert_series_equal(res1, pd.Series([False, False, False, True, True, True,
                                         True, True, False, False]))
    assert_series_equal(res2, pd.Series([False, False, True, True, True, False,
                                         False, True, False, False]))
    assert_series_equal(res3, pd.Series([False, False, False, False, False,
                                         False, False, True, False, False]))
    assert not all(res4)
    assert_series_equal(res5, pd.Series([False, False, False, False, True,
                                         False, False, False, False, False]))
    assert_series_equal(res6, pd.Series(index=stale_data[1:].index,
                                        data=[False, False, True, True, True,
                                              True, True, False, False]))
    assert_series_equal(res7, pd.Series(index=stale_data[1:8].index,
                                        data=[False, False, True, True, True,
                                              True, True]))


def test_detect_stale_values_handles_negatives(data_with_negatives):
    """detect_stale_values works with negative values."""
    res = gaps.detect_stale_values(data_with_negatives)
    assert_series_equal(res, pd.Series([False, False, True, True, False, False,
                                        False]))
    res = gaps.detect_stale_values(data_with_negatives, atol=1e-3)
    assert_series_equal(res, pd.Series([False, False, True, True, True, True,
                                        True]))
    res = gaps.detect_stale_values(data_with_negatives, atol=1e-5)
    assert_series_equal(res, pd.Series([False, False, True, True, True, False,
                                        False]))
    res = gaps.detect_stale_values(data_with_negatives, atol=2e-5)
    assert_series_equal(res, pd.Series([False, False, True, True, True, True,
                                        True]))


def test_detect_stale_values_raises_error(stale_data):
    """detect_stale_values raises a ValueError for 'window' < 2."""
    with pytest.raises(ValueError):
        gaps.detect_stale_values(stale_data, window=1)


@pytest.fixture
def interpolated_data():
    """A series that contains linear interpolation."""
    data = [1.0, 1.001, 1.002001, 1.003, 1.004, 1.001001, 1.001001, 1.001001,
            1.2, 1.3, 1.5, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]
    return pd.Series(data=data)


def test_detect_interpolation(interpolated_data):
    """Interpolation is detected correclty."""
    res1 = gaps.detect_interpolation(interpolated_data)
    assert_series_equal(res1, pd.Series([False, False, False, False, False,
                                         False, False, True, False, False,
                                         False, False, False, True, True, True,
                                         False]))
    res2 = gaps.detect_interpolation(interpolated_data, rtol=1e-2)
    assert_series_equal(res2, pd.Series([False, False, True, True, True,
                                         False, False, True, False, False,
                                         False, False, False, True, True, True,
                                         False]))
    res3 = gaps.detect_interpolation(interpolated_data, window=5)
    assert_series_equal(res3, pd.Series([False, False, False, False, False,
                                         False, False, False, False, False,
                                         False, False, False, False, False,
                                         True, False]))
    res4 = gaps.detect_interpolation(interpolated_data, atol=1e-2)
    assert_series_equal(res4, pd.Series([False, False, True, True, True,
                                         True, True, True, False, False,
                                         False, False, False, True, True, True,
                                         False]))


def test_detect_interpolation_handles_negatives(data_with_negatives):
    """Interpolation is detected correctly when data contains negatives."""
    res = gaps.detect_interpolation(data_with_negatives, atol=1e-5)
    assert_series_equal(res, pd.Series([False, False, True, True, True, True,
                                        False]))
    res = gaps.detect_stale_values(data_with_negatives, atol=1e-4)
    assert_series_equal(res, pd.Series([False, False, True, True, True, True,
                                        True]))


def test_detect_interpolation_raises_error(interpolated_data):
    """detect_interpolation raises a ValueError for 'window' < 3."""
    with pytest.raises(ValueError):
        gaps.detect_interpolation(interpolated_data, window=2)
