"""Tests for the quality.outliers module."""
from pandas.testing import assert_series_equal
import pandas as pd
from pvanalytics.quality import outliers


def test_tukey_no_outliers():
    """If all the data has the same value there are no outliers"""
    data = pd.Series([1 for _ in range(10)])
    assert_series_equal(
        pd.Series([True for _ in range(10)]),
        outliers.tukey(data)
    )


def test_tukey_outlier_below():
    """tukey properly detects an outlier that is too low."""
    data = pd.Series([5, 9, 9, 8, 7, 1, 7, 8, 6, 8])
    assert_series_equal(
        pd.Series([True, True, True, True, True,
                   False, True, True, True, True]),
        outliers.tukey(data)
    )


def test_tukey_outlier_above():
    """tukey properly detects an outlier that is too high."""
    data = pd.Series([0, 1, 3, 2, 1, 3, 4, 1, 1, 9, 2])
    assert_series_equal(
        pd.Series([True, True, True, True, True, True,
                   True, True, True, False, True]),
        outliers.tukey(data)
    )


def test_tukey_lower_criteria():
    """With lower criteria the single small value is not an outlier."""
    data = pd.Series([5, 9, 9, 8, 7, 1, 7, 8, 6, 8])
    assert_series_equal(
        pd.Series([True for _ in range(len(data))]),
        outliers.tukey(data, k=3)
    )
