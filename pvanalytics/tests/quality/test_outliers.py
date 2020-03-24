"""Tests for the quality.outliers module."""
from pandas.testing import assert_series_equal
import pandas as pd
from pvanalytics.quality import outliers


def test_iqr_no_outliers():
    """If all the data has the same value there are no outliers"""
    data = pd.Series([1 for _ in range(10)])
    assert_series_equal(
        pd.Series([True for _ in range(10)]),
        outliers.filter_iqr(data)
    )


def test_iqr_one_outlier():
    """filter_iqr properly detects an outlier."""
    data = pd.Series([5, 9, 9, 8, 7, 1, 7, 8, 6, 8])
    assert_series_equal(
        pd.Series([True, True, True, True, True,
                   False, True, True, True, True]),
        outliers.filter_iqr(data)
    )


def test_iqr_lower_criteria():
    """With lower criteria the single small value is not an outlier."""
    data = pd.Series([5, 9, 9, 8, 7, 1, 7, 8, 6, 8])
    assert_series_equal(
        pd.Series([True for _ in range(len(data))]),
        outliers.filter_iqr(data, k=3)
    )
