"""Tests for gaps quality control functions."""
import pytest
import pandas as pd
import numpy as np
from pandas.util.testing import assert_series_equal
from pvanalytics.quality import gaps


@pytest.fixture
def stale_data():
    """A series that contains stuck values.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    data = [1.0, 1.001, 1.001, 1.001, 1.001, 1.001001, 1.001, 1.001, 1.2, 1.3]
    return pd.Series(data=data)


@pytest.fixture
def data_with_negatives():
    """A series that contains stuck values, interpolation, and negatives.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    data = [0.0, 0.0, 0.0, -0.0, 0.00001, 0.000010001, -0.00000001]
    return pd.Series(data=data)


def test_stale_values_diff(stale_data):
    """stale_values_diff properly identifies stuck values.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    res1 = gaps.stale_values_diff(stale_data)
    res2 = gaps.stale_values_diff(stale_data, rtol=1e-8, window=2)
    res3 = gaps.stale_values_diff(stale_data, window=7)
    res4 = gaps.stale_values_diff(stale_data, window=8)
    res5 = gaps.stale_values_diff(stale_data, rtol=1e-8, window=4)
    res6 = gaps.stale_values_diff(stale_data[1:])
    res7 = gaps.stale_values_diff(stale_data[1:8])
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


def test_stale_values_diff_handles_negatives(data_with_negatives):
    """stale_values_diff works with negative values.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    res = gaps.stale_values_diff(data_with_negatives)
    assert_series_equal(res, pd.Series([False, False, True, True, False, False,
                                        False]))
    res = gaps.stale_values_diff(data_with_negatives, atol=1e-3)
    assert_series_equal(res, pd.Series([False, False, True, True, True, True,
                                        True]))
    res = gaps.stale_values_diff(data_with_negatives, atol=1e-5)
    assert_series_equal(res, pd.Series([False, False, True, True, True, False,
                                        False]))
    res = gaps.stale_values_diff(data_with_negatives, atol=2e-5)
    assert_series_equal(res, pd.Series([False, False, True, True, True, True,
                                        True]))


def test_stale_values_diff_raises_error(stale_data):
    """stale_values_diff raises a ValueError for 'window' < 2.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    with pytest.raises(ValueError):
        gaps.stale_values_diff(stale_data, window=1)


@pytest.fixture
def interpolated_data():
    """A series that contains linear interpolation.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    data = [1.0, 1.001, 1.002001, 1.003, 1.004, 1.001001, 1.001001, 1.001001,
            1.2, 1.3, 1.5, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]
    return pd.Series(data=data)


def test_interpolation_diff(interpolated_data):
    """Interpolation is detected correclty.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    res1 = gaps.interpolation_diff(interpolated_data)
    assert_series_equal(res1, pd.Series([False, False, False, False, False,
                                         False, False, True, False, False,
                                         False, False, False, True, True, True,
                                         False]))
    res2 = gaps.interpolation_diff(interpolated_data, rtol=1e-2)
    assert_series_equal(res2, pd.Series([False, False, True, True, True,
                                         False, False, True, False, False,
                                         False, False, False, True, True, True,
                                         False]))
    res3 = gaps.interpolation_diff(interpolated_data, window=5)
    assert_series_equal(res3, pd.Series([False, False, False, False, False,
                                         False, False, False, False, False,
                                         False, False, False, False, False,
                                         True, False]))
    res4 = gaps.interpolation_diff(interpolated_data, atol=1e-2)
    assert_series_equal(res4, pd.Series([False, False, True, True, True,
                                         True, True, True, False, False,
                                         False, False, False, True, True, True,
                                         False]))


def test_interpolation_diff_handles_negatives(data_with_negatives):
    """Interpolation is detected correctly when data contains negatives.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    res = gaps.interpolation_diff(data_with_negatives, atol=1e-5)
    assert_series_equal(res, pd.Series([False, False, True, True, True, True,
                                        False]))
    res = gaps.stale_values_diff(data_with_negatives, atol=1e-4)
    assert_series_equal(res, pd.Series([False, False, True, True, True, True,
                                        True]))


def test_interpolation_diff_raises_error(interpolated_data):
    """interpolation raises a ValueError for 'window' < 3.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    with pytest.raises(ValueError):
        gaps.interpolation_diff(interpolated_data, window=2)


def test_valid_between_no_missing_data():
    """If there is no missing data firstlastvaliddays should return the
    start and end of the series.

    """
    index = pd.date_range(
        freq='15T',
        start='01-01-2020',
        end='08-01-2020 23:00'
    )
    series = pd.Series(
        data=np.full(len(index), 10),
        index=index
    )
    firstvalid, lastvalid = gaps.valid_between(series)
    assert firstvalid.date() == pd.Timestamp('01-01-2020').date()
    assert lastvalid.date() == pd.Timestamp('08-01-2020').date()


def test_first_day_missing_data():
    """If the first day is missing data, the first valid date should be
    the second day.

    """
    index = pd.date_range(
        freq='15T',
        start='01-01-2020',
        end='08-01-2020 23:00'
    )
    data = np.full(len(index), 10)
    series = pd.Series(data=data, index=index)
    series['01-01-2020 00:00':'01-02-2020 00:00'] = np.nan
    firstvalid, lastvalid = gaps.valid_between(series)
    assert firstvalid.date() == pd.Timestamp('01-02-2020').date()
    assert lastvalid.date() == pd.Timestamp('08-01-2020').date()


def test_first_and_fifth_days_missing():
    """First valid date should be the sixth of January."""
    index = pd.date_range(
        freq='15T',
        start='01-01-2020',
        end='08-01-2020 23:00'
    )
    data = np.full(len(index), 10)
    series = pd.Series(data=data, index=index)
    series['01-01-2020 00:00':'01-02-2020 00:00'] = np.nan
    series['01-05-2020 00:00':'01-06-2020 00:00'] = np.nan
    firstvalid, lastvalid = gaps.valid_between(series)
    assert firstvalid.date() == pd.Timestamp('01-06-2020').date()
    assert lastvalid.date() == pd.Timestamp('08-01-2020').date()


def test_last_two_days_missing():
    """If the last two days of data are missing last valid day should be
    July 30.

    """
    index = pd.date_range(
        freq='15T',
        start='01-01-2020',
        end='08-01-2020 23:00'
    )
    data = np.full(len(index), 10)
    series = pd.Series(data=data, index=index)
    series['07-31-2020 00:00':'08-01-2020 23:00'] = np.nan
    firstvalid, lastvalid = gaps.valid_between(series)
    assert firstvalid.date() == pd.Timestamp('01-01-2020').date()
    assert lastvalid.date() == pd.Timestamp('07-30-2020').date()


def test_valid_between_no_data():
    """If the passed to valid_between is empty the returns (None, None)."""
    index = pd.date_range(
        freq='15T',
        start='01-01-2020',
        end='08-01-2020 23:00'
    )
    series = pd.Series(index=index, data=np.full(len(index), np.nan))
    assert (None, None) == gaps.valid_between(series)


def test_valid_between_sparse_data():
    """Check that days with only a few hours of data aren't considered
    valid.

    """
    index = pd.date_range(
        freq='15T',
        start='01-01-2020',
        end='08-01-2020 23:00'
    )
    series = pd.Series(index=index, data=np.full(len(index), 2.3))
    series['01-02-2020 00:00':'01-02-2020 06:00'] = np.nan
    series['01-02-2020 08:00':'01-02-2020 21:00'] = np.nan
    series['07-31-2020 07:00':] = np.nan
    start, end = gaps.valid_between(series)
    assert start.date() == pd.Timestamp('01-03-2020').date()
    assert end.date() == pd.Timestamp('07-30-2020').date()


def test_valid_between_not_enough_data():
    """Only one day of data is not ehough for any valid days."""
    index = pd.date_range(
        freq='15T',
        start='01-01-2020',
        end='08-01-2020 23:00'
    )
    series = pd.Series(index=index, dtype='float64')
    series['02-23-2020 08:00':'02-24-2020 08:00'] = 1
    assert (None, None) == gaps.valid_between(series)


def test_valid_between_one_day():
    """Works when there is exactly the minimum number of consecutive
    days with data.

    """
    index = pd.date_range(
        freq='15T',
        start='01-01-2020',
        end='08-01-2020 23:00'
    )
    series = pd.Series(index=index, dtype='float64')
    series['05-05-2020'] = 2
    start, end = gaps.valid_between(series, days=1)
    assert start.date() == pd.Timestamp('05-05-2020').date()
    assert end.date() == pd.Timestamp('05-05-2020').date()


def test_valid_between_with_gaps_in_middle():
    """When there are gaps in the data longer than `days` valid between
    should include those gaps, as long as there are `days` consecutive
    days with enough data some time after the gap.

    """
    index = pd.date_range(
        freq='15T',
        start='01-01-2020',
        end='08-01-2020 23:00'
    )
    series = pd.Series(index=index, data=np.full(len(index), 1))
    series['03-05-2020':'03-25-2020'] = np.nan
    start, end = gaps.valid_between(series, days=5)
    assert start.date() == index[0].date()
    assert end.date() == index[-1].date()


def test_trim():
    index = pd.date_range(
        freq='15T',
        start='01-01-2020',
        end='08-01-2020 23:00'
    )
    series = pd.Series(index=index, data=np.full(len(index), 1))
    series['01-02-2020':'01-07-2020 13:00'] = np.nan
    series['01-10-2020':'01-11-2020'] = np.nan
    assert_series_equal(
        series[gaps.trim(series, days=3)],
        series['01-07-2020':'08-01-2020 00:00']
    )
