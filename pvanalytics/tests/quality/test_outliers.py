"""Tests for the quality.outliers module."""
import pandas as pd
import numpy as np
from pandas.util.testing import assert_series_equal
from pvanalytics.quality import outliers


def test_tukey_no_outliers():
    """If all the data has the same value there are no outliers"""
    data = pd.Series([1 for _ in range(10)])
    assert_series_equal(
        pd.Series([False for _ in range(10)]),
        outliers.tukey(data)
    )


def test_tukey_outlier_below():
    """tukey properly detects an outlier that is too low."""
    data = pd.Series([5, 9, 9, 8, 7, 1, 7, 8, 6, 8])
    assert_series_equal(
        pd.Series([False, False, False, False, False,
                   True, False, False, False, False]),
        outliers.tukey(data)
    )


def test_tukey_outlier_above():
    """tukey properly detects an outlier that is too high."""
    data = pd.Series([0, 1, 3, 2, 1, 3, 4, 1, 1, 9, 2])
    assert_series_equal(
        pd.Series([False, False, False, False, False, False,
                   False, False, False, True, False]),
        outliers.tukey(data)
    )


def test_tukey_lower_criteria():
    """With lower criteria the single small value is not an outlier."""
    data = pd.Series([5, 9, 9, 8, 7, 1, 7, 8, 6, 8])
    assert_series_equal(
        pd.Series([False for _ in range(len(data))]),
        outliers.tukey(data, k=3)
    )


def test_zscore_all_same():
    """If all data is identical there are no outliers."""
    data = pd.Series([1 for _ in range(20)])
    np.seterr(invalid='ignore')
    assert_series_equal(
        pd.Series([False for _ in range(20)]),
        outliers.zscore(data)
    )
    np.seterr(invalid='warn')


def test_zscore_outlier_above():
    """Correctly idendifies an outlier above the mean."""
    data = pd.Series([1, 0, -1, 0, 1, -1, 10])
    assert_series_equal(
        pd.Series([False, False, False, False, False, False, True]),
        outliers.zscore(data)
    )


def test_zscore_outlier_below():
    """Correctly idendifies an outlier below the mean."""
    data = pd.Series([1, 0, -1, 0, 1, -1, -10])
    assert_series_equal(
        pd.Series([False, False, False, False, False, False, True]),
        outliers.zscore(data)
    )


def test_zscore_zmax():
    """Increasing zmax excludes outliers closest to the mean."""
    data = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 10])
    assert_series_equal(
        data[-2:],
        data[outliers.zscore(data)]
    )
    assert_series_equal(
        data[-1:],
        data[outliers.zscore(data, zmax=3)]
    )
    assert (~outliers.zscore(data, zmax=5)).all()
