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


def test_start_stop_dates_no_missing_data():
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
    firstvalid, lastvalid = gaps.start_stop_dates(series)
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
    firstvalid, lastvalid = gaps.start_stop_dates(series)
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
    firstvalid, lastvalid = gaps.start_stop_dates(series)
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
    firstvalid, lastvalid = gaps.start_stop_dates(series)
    assert firstvalid.date() == pd.Timestamp('01-01-2020').date()
    assert lastvalid.date() == pd.Timestamp('07-30-2020').date()


def test_start_stop_dates_no_data():
    """If the passed to start_stop_dates is empty the returns (None, None)."""
    index = pd.date_range(
        freq='15T',
        start='01-01-2020',
        end='08-01-2020 23:00'
    )
    series = pd.Series(index=index, data=np.full(len(index), np.nan))
    assert (None, None) == gaps.start_stop_dates(series)


def test_start_stop_dates_sparse_data():
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
    start, end = gaps.start_stop_dates(series)
    assert start.date() == pd.Timestamp('01-03-2020').date()
    assert end.date() == pd.Timestamp('07-30-2020').date()


def test_start_stop_dates_not_enough_data():
    """Only one day of data is not ehough for any valid days."""
    index = pd.date_range(
        freq='15T',
        start='01-01-2020',
        end='08-01-2020 23:00'
    )
    series = pd.Series(index=index, dtype='float64')
    series['02-23-2020 08:00':'02-24-2020 08:00'] = 1
    assert (None, None) == gaps.start_stop_dates(series)


def test_start_stop_dates_one_day():
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
    start, end = gaps.start_stop_dates(series, days=1)
    assert start.date() == pd.Timestamp('05-05-2020').date()
    assert end.date() == pd.Timestamp('05-05-2020').date()


def test_start_stop_dates_with_gaps_in_middle():
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
    start, end = gaps.start_stop_dates(series, days=5)
    assert start.date() == index[0].date()
    assert end.date() == index[-1].date()


def test_trim():
    """gaps.trim() should return a boolean mask that selects only the good
    data in the middle of a series.

    """
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


def test_trim_empty():
    """gaps.trim() returns all False for series with no valid days."""
    index = pd.date_range(
        freq='15T',
        start='01-01-2020',
        end='08-01-2020 23:00'
    )
    series = pd.Series(index=index, dtype='float64')
    series.iloc[::(24*60)] = 1
    assert (~gaps.trim(series, days=3)).all()


def test_daily_completeness_all_nans():
    """A data set with all nans has completeness 0 for each day."""
    completeness = gaps.daily_completeness(
        pd.Series(
            np.nan,
            index=pd.date_range('01/01/2020 00:00', freq='1H', periods=48),
            dtype='float64'
        )
    )
    assert_series_equal(
        pd.Series(
            0.0,
            index=pd.date_range(start='01/01/2020', freq='D', periods=2)
        ),
        completeness
    )


def test_daily_completeness_no_data():
    """A data set with completely missing timestamps and NaNs has
    completeness 0."""
    four_days = pd.date_range(start='01/01/2020', freq='D', periods=4)
    completeness = gaps.daily_completeness(
        pd.Series(index=four_days, dtype='float64'), freq='15T'
    )
    assert_series_equal(
        pd.Series(0.0, index=four_days),
        completeness
    )


def test_daily_completeness_incomplete_index():
    """A series with one data point per hour has 25% completeness at
    15-minute sample frequency"""
    data = pd.Series(
        1,
        index=pd.date_range(start='01/01/2020', freq='1H', periods=72),
    )
    completeness = gaps.daily_completeness(data, freq='15T')
    assert_series_equal(
        pd.Series(
            0.25,
            index=pd.date_range(start='01/01/2020', freq='D', periods=3)
        ),
        completeness
    )


def test_daily_completeness_complete():
    """A series with data at every point has completeness 1.0"""
    data = pd.Series(
        1, index=pd.date_range(start='01/01/2020', freq='15T', periods=24*4*2)
    )
    completeness = gaps.daily_completeness(data)
    assert_series_equal(
        pd.Series(
            1.0,
            index=pd.date_range(start='01/01/2020', freq='D', periods=2)
        ),
        completeness
    )


def test_daily_completeness_freq_too_high():
    """If the infered freq is shorter than the passed freq an exception is
    raised."""
    data = pd.Series(
        1,
        index=pd.date_range(start='1/1/2020', freq='15T', periods=24*4*4)
    )
    with pytest.raises(ValueError):
        gaps.daily_completeness(data, freq='16T')
    with pytest.raises(ValueError):
        gaps.daily_completeness(data, freq='1H')


def test_complete_threshold_zero():
    """minimum_completeness of 0 returns all True regardless of data."""
    ten_days = pd.date_range(
        start='01/01/2020', freq='15T', end='1/10/2020', closed='left')
    data = pd.Series(index=ten_days, dtype='float64')
    assert_series_equal(
        pd.Series(True, index=data.index),
        gaps.complete(data, minimum_completeness=0)
    )
    data[pd.date_range(
        start='01/01/2020', freq='1D', end='1/10/2020', closed='left')] = 1.0
    data.dropna()
    assert_series_equal(
        pd.Series(True, index=data.index),
        gaps.complete(data, minimum_completeness=0, freq='15T')
    )
    data = pd.Series(1.0, index=ten_days)
    assert_series_equal(
        pd.Series(True, index=data.index),
        gaps.complete(data, minimum_completeness=0)
    )


def test_complete_threshold_one():
    """If minimum_completeness=1 then any missing data on a day means all
    data for the day is flagged False."""
    ten_days = pd.date_range(
        start='01/01/2020', freq='15T', end='01/10/2020', closed='left')
    data = pd.Series(index=ten_days, dtype='float64')
    assert_series_equal(
        pd.Series(False, index=data.index),
        gaps.complete(data, minimum_completeness=1.0)
    )
    data.loc[:] = 1
    assert_series_equal(
        pd.Series(True, index=data.index),
        gaps.complete(data, minimum_completeness=1.0)
    )
    # remove one data-point per day
    days = pd.date_range(
        start='1/1/2020', freq='1D', end='1/10/2020', closed='left')
    data.loc[days] = np.nan
    assert_series_equal(
        pd.Series(False, index=data.index),
        gaps.complete(data, minimum_completeness=1.0)
    )
    # check that dropping the NaNs still gives the same result with
    # and without passing `freq`. (There should be enough data to infer the
    # correct frequency if only one value is missing on each day.)
    data.dropna()
    assert_series_equal(
        pd.Series(False, index=data.index),
        gaps.complete(data, minimum_completeness=1.0)
    )
    assert_series_equal(
        gaps.complete(data, minimum_completeness=1.0),
        gaps.complete(data, minimum_completeness=1.0, freq='15T')
    )


def test_complete():
    """Test gaps.complete with varying amounts of missing data."""
    ten_days = pd.date_range(
        start='1/1/2020', freq='H', end='1/10/2020', closed='left')
    data = pd.Series(index=ten_days, dtype='float64')
    data.loc['1/1/2020'] = 1.0
    day_two_values = pd.date_range(
        start='1/2/2020', freq='2H', end='1/3/2020', closed='left')
    data.loc[day_two_values] = 2.0
    day_three_values = pd.date_range(
        start='1/3/2020', freq='3H', end='1/4/2020', closed='left')
    data.loc[day_three_values] = 3.0
    day_four_values = pd.date_range(
        start='1/4/2020', freq='4H', end='1/5/2020', closed='left')
    data.loc[day_four_values] = 4.0
    data.loc['1/5/2020':] = 5.0

    expected = pd.Series(False, index=data.index)
    expected.loc['1/1/2020'] = True
    expected.loc['1/5/2020':] = True
    assert_series_equal(
        expected,
        gaps.complete(data, minimum_completeness=1.0)
    )

    expected.loc['1/2/2020'] = True
    assert_series_equal(
        expected,
        gaps.complete(data, minimum_completeness=0.5)
    )

    expected.loc['1/3/2020'] = True
    assert_series_equal(
        expected,
        gaps.complete(data, minimum_completeness=0.3)
    )

    assert_series_equal(
        pd.Series(True, index=data.index),
        gaps.complete(data, minimum_completeness=0.2)
    )
