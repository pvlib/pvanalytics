"""Tests for the quality.outliers module."""
import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal
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


def test_zscore_raise_nan_input():
    data = pd.Series([1, 0, -1, 0, np.nan, 1, -1, 10])

    with pytest.raises(ValueError):
        outliers.zscore(data, nan_policy='raise')


def test_zscore_invalid_nan_policy():
    data = pd.Series([1, 0, -1, 0, np.nan, 1, -1, 10])

    with pytest.raises(ValueError):
        outliers.zscore(data, nan_policy='incorrect_str')


def test_zscore_omit_nan_input():
    data = pd.Series([1, 0, -1, 0, np.nan, 1, -1, 10])
    assert_series_equal(
        pd.Series([False, False, False, False, False, False, False, True]),
        outliers.zscore(outliers.zscore(data, nan_policy='omit'))
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


def test_hampel_all_same():
    """outliers.hampel identifies no outlier if all data is the same."""
    data = pd.Series(1, index=range(0, 50))
    assert_series_equal(
        outliers.hampel(data),
        pd.Series(False, index=range(0, 50))
    )


def test_hampel_one_outlier():
    """If all data is same but one value outliers.hampel should identify
    that value as an outlier."""
    np.random.seed(1000)
    data = pd.Series(np.random.uniform(0, 1, size=50))
    data.iloc[20] = 10
    expected = pd.Series(False, index=data.index)
    expected.iloc[20] = True
    assert_series_equal(
        outliers.hampel(data, window=11),
        expected
    )


def test_hampel_max_deviation():
    """Increasing max_deviation causes fewer values to be identified as
    outliers."""
    np.random.seed(1000)
    data = pd.Series(np.random.uniform(-1, 1, size=100))
    data.iloc[20] = -25
    data.iloc[40] = 15
    data.iloc[60] = 5

    expected = pd.Series(False, index=data.index)
    expected.iloc[[20, 40, 60]] = True

    assert_series_equal(
        data[outliers.hampel(data, window=11)],
        data[expected]
    )

    expected.iloc[60] = False
    assert_series_equal(
        data[outliers.hampel(data, window=11, max_deviation=15)],
        data[expected]
    )

    expected.iloc[40] = False
    assert_series_equal(
        data[outliers.hampel(data, window=11, max_deviation=16)],
        data[expected]
    )


def test_hampel_scale():
    np.random.seed(1000)
    data = pd.Series(np.random.uniform(-1, 1, size=100))
    data.iloc[20] = -25
    data.iloc[40] = 15
    data.iloc[60] = 5
    assert not (outliers.hampel(data) == outliers.hampel(data, scale=0.1)).all()


def test_compare_reference_difference():
    """compare_reference identifies outliers in the difference."""
    actual = pd.Series([1.0, 2.0, 3.0, 4.0, 10.0])
    reference = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    # deviations: [0, 0, 0, 0, 5]
    # z-scores: [-0.5, -0.5, -0.5, -0.5, 2.0]
    expected = pd.Series([False, False, False, False, True])
    assert_series_equal(
        outliers.compare_reference(
            actual, reference,
            comparison='difference',
            method='zscore',
            zmax=1.5
        ),
        expected
    )


def test_compare_reference_relative():
    """compare_reference identifies outliers in the relative difference."""
    actual = pd.Series([1.0, 2.0, 3.0, 4.0, 8.0])
    reference = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    # deviations: [0, 0, 0, 0, 0.6]
    # z-scores: [-0.5, -0.5, -0.5, -0.5, 2.0]
    expected = pd.Series([False, False, False, False, True])
    assert_series_equal(
        outliers.compare_reference(
            actual, reference,
            comparison='relative',
            method='zscore',
            zmax=1.5
        ),
        expected
    )


def test_compare_reference_absolute_difference():
    """compare_reference identifies outliers in the absolute difference."""
    actual = pd.Series([1.0, 2.0, 3.0, 4.0, 0.0])
    reference = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    # deviations: [0, 0, 0, 0, 5]
    expected = pd.Series([False, False, False, False, True])
    assert_series_equal(
        outliers.compare_reference(
            actual, reference,
            comparison='absolute_difference',
            method='zscore',
            zmax=1.5
        ),
        expected
    )


def test_compare_reference_tukey():
    """compare_reference works with tukey method."""
    actual = pd.Series([1.0, 2.0, 3.0, 4.0, 20.0])
    reference = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    # deviations: [0, 0, 0, 0, 15]
    expected = pd.Series([False, False, False, False, True])
    assert_series_equal(
        outliers.compare_reference(
            actual, reference,
            method='tukey'
        ),
        expected
    )


def test_quantile_threshold():
    """quantile_threshold returns expected values."""
    x = np.arange(100)
    # y = 2x + 10 + noise
    np.random.seed(1234)
    y = 2 * x + 10 + np.random.normal(0, 1, 100)
    threshold = outliers.quantile_threshold(x, pd.Series(y), 0.9)
    assert len(threshold) == 100
    # For a large enough sample, the slope should be close to 2
    # and intercept close to 10 + 1.28*1
    # We just check if it's generally in the right ballpark
    assert np.all(threshold > 2 * x)
    assert np.all(threshold < 2 * x + 20)

