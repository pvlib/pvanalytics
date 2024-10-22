"""Tests for :py:mod:`features.daytime`"""
import pytest
import pandas as pd
import numpy as np
import pvlib
from pvlib.location import Location
from datetime import date
from pvanalytics.features import daytime
from ..conftest import DATA_DIR

test_file_1 = DATA_DIR / "serf_east_1min_ac_power.csv"


@pytest.fixture(scope='module',
                params=['h', '15min',
                        pytest.param('min', marks=pytest.mark.slow)])
def clearsky_january(request, albuquerque):
    return albuquerque.get_clearsky(
        pd.date_range(
            start='1/1/2020',
            end='1/30/2020',
            tz='MST',
            freq=request.param
        ),
        model='simplified_solis'
    )


@pytest.fixture
def ac_power_series():
    # Pull down the saved PVLib dataframe and process it
    time_series = pd.read_csv(test_file_1,
                              parse_dates=True,
                              index_col=0).squeeze()
    return time_series


@pytest.fixture
def modeled_midday_series(ac_power_series):
    # Get the modeled sunrise and sunset for the location
    dates = ac_power_series.index.normalize().unique()
    modeled_sunrise_sunset_df = pvlib.solarposition.sun_rise_set_transit_spa(
        dates, 39.742, -105.1727)
    # Take the 'transit' column as the midday point between sunrise and
    # sunset for each day in the modeled irradiance series
    modeled_midday_series = modeled_sunrise_sunset_df['transit']
    modeled_midday_series.index = dates.date
    return modeled_midday_series


@pytest.fixture
def daytime_mask_left_aligned(ac_power_series):
    # Resample the time series to 5-minute left aligned intervals
    ac_power_series_left = ac_power_series.resample('5min',
                                                    label='left').mean()
    data_freq = pd.infer_freq(ac_power_series_left.index)
    daytime_mask = daytime.power_or_irradiance(ac_power_series_left,
                                               freq=data_freq)
    return daytime_mask


@pytest.fixture
def daytime_mask_right_aligned(ac_power_series):
    # Resample the time series to 5-minute right aligned intervals. Lop off the
    # last entry as it is moved to the next day (3/20)
    ac_power_series_right = ac_power_series.resample('5min',
                                                     label='right').mean()[:-1]
    data_freq = pd.infer_freq(ac_power_series_right.index)
    daytime_mask = daytime.power_or_irradiance(ac_power_series_right,
                                               freq=data_freq)
    return daytime_mask


@pytest.fixture
def daytime_mask_center_aligned(ac_power_series):
    # Resample the time series to 5-minute center aligned intervals (take
    # left alignment and shift by frequency/2)
    ac_power_series_center = ac_power_series.resample('5min',
                                                      label='left').mean()
    ac_power_series_center.index = (ac_power_series_center.index +
                                    (pd.Timedelta("5min") / 2))
    data_freq = pd.infer_freq(ac_power_series_center.index)
    daytime_mask = daytime.power_or_irradiance(ac_power_series_center,
                                               freq=data_freq)
    return daytime_mask


def _assert_daytime_no_shoulder(clearsky, output):
    # every night-time value in `output` has low or 0 irradiance
    assert all(clearsky[~output] < 3)
    if pd.infer_freq(clearsky.index) in ['T', 'min']:
        # Blur the boundaries between night and day if testing
        # high-frequency data since the daytime filtering algorithm does
        # not have one-minute accuracy.
        clearsky = clearsky.rolling(window=30, center=True).max()
    # every day-time value is within 15 minutes of a non-zero
    # irradiance measurement
    assert all(clearsky[output] > 0)


def test_daytime_with_clipping(clearsky_january):
    ghi = clearsky_january['ghi'].copy()
    ghi.loc[ghi >= 500] = 500
    _assert_daytime_no_shoulder(
        clearsky_january['ghi'],
        daytime.power_or_irradiance(ghi)
    )
    # Include a period where data goes to zero during clipping and
    # returns to normal after the clipping is done
    ghi.loc[ghi['1/3/2020'].between_time('12:30', '15:30').index] = 0
    _assert_daytime_no_shoulder(
        clearsky_january['ghi'],
        daytime.power_or_irradiance(ghi)
    )


def test_daytime_overcast(clearsky_january):
    ghi = clearsky_january['ghi'].copy()
    ghi.loc['1/3/2020':'1/5/2020'] *= 0.5
    ghi.loc['1/7/2020':'1/8/2020'] *= 0.6
    _assert_daytime_no_shoulder(
        clearsky_january['ghi'],
        daytime.power_or_irradiance(ghi)
    )


def test_daytime_split_day():
    location = Location(35, -150)
    # no tz:
    times = pd.date_range(start='1/1/2020', end='1/10/2020', freq='15min')
    clearsky = location.get_clearsky(times, model='simplified_solis')
    _assert_daytime_no_shoulder(
        clearsky['ghi'],
        daytime.power_or_irradiance(clearsky['ghi'])
    )


def test_daytime(clearsky_january):
    _assert_daytime_no_shoulder(
        clearsky_january['ghi'],
        daytime.power_or_irradiance(clearsky_january['ghi'])
    )
    # punch a mid-day hole
    ghi = clearsky_january['ghi'].copy()
    ghi.loc['1/10/2020 12:00':'1/10/2020 14:00'] = 0
    _assert_daytime_no_shoulder(
        clearsky_january['ghi'],
        daytime.power_or_irradiance(ghi)
    )


def test_daytime_daylight_savings(albuquerque):
    spring = pd.date_range(
        start='2/10/2020',
        end='4/10/2020',
        freq='15min',
        tz='America/Denver'
    )
    clearsky_spring = albuquerque.get_clearsky(
        spring,
        model='simplified_solis'
    )
    _assert_daytime_no_shoulder(
        clearsky_spring['ghi'],
        daytime.power_or_irradiance(clearsky_spring['ghi'])
    )
    fall = pd.date_range(
        start='10/1/2020',
        end='12/1/2020',
        freq='15min',
        tz='America/Denver'
    )
    clearsky_fall = albuquerque.get_clearsky(
        fall,
        model='simplified_solis'
    )
    _assert_daytime_no_shoulder(
        clearsky_fall['ghi'],
        daytime.power_or_irradiance(clearsky_fall['ghi'])
    )


def test_daytime_zero_at_end_of_day(clearsky_january):
    ghi = clearsky_january['ghi'].copy()
    ghi.loc['1/5/2020 16:00':'1/6/2020 00:00'] = 0
    _assert_daytime_no_shoulder(
        clearsky_january['ghi'],
        daytime.power_or_irradiance(ghi)
    )
    # test a period of zeros starting earlier in the day
    ghi.loc['1/5/2020 12:00':'1/6/2020 00:00'] = 0
    _assert_daytime_no_shoulder(
        clearsky_january['ghi'],
        daytime.power_or_irradiance(ghi)
    )


def test_daytime_zero_at_start_of_day(clearsky_january):
    ghi = clearsky_january['ghi'].copy()
    ghi.loc['1/5/2020 00:00':'1/5/2020 09:00'] = 0
    _assert_daytime_no_shoulder(
        clearsky_january['ghi'],
        daytime.power_or_irradiance(ghi)
    )


def test_daytime_outliers(clearsky_january):
    outliers = pd.Series(False, index=clearsky_january.index)
    outliers.loc['1/5/2020 13:00'] = True
    outliers.loc['1/5/2020 14:00'] = True
    outliers.loc['1/10/2020 02:00'] = True
    ghi = clearsky_january['ghi'].copy()
    ghi.loc['1/5/2020 1300'] = 999
    ghi.loc['1/5/2020 14:00'] = -999
    ghi.loc['1/10/2020 02:00'] = -999
    _assert_daytime_no_shoulder(
        clearsky_january['ghi'],
        daytime.power_or_irradiance(clearsky_january['ghi'], outliers=outliers)
    )


def test_daytime_missing_data(clearsky_january):
    ghi = clearsky_january['ghi'].copy()
    ghi.loc['1/5/2020 16:00':'1/6/2020 11:30'] = np.nan
    # test with NaNs
    _assert_daytime_no_shoulder(
        clearsky_january['ghi'],
        daytime.power_or_irradiance(ghi)
    )
    # test with completely missing data
    ghi.dropna(inplace=True)
    _assert_daytime_no_shoulder(
        ghi,
        daytime.power_or_irradiance(
            ghi, freq=pd.infer_freq(clearsky_january.index)
        )
    )


def test_daytime_variable(clearsky_january):
    ghi = clearsky_january['ghi'].copy()
    np.random.seed(1337)
    ghi.loc['1/10/2020'] *= np.random.rand(len(ghi['1/10/2020']))
    ghi.loc['1/11/2020'] *= np.random.rand(len(ghi['1/11/2020']))
    _assert_daytime_no_shoulder(
        clearsky_january['ghi'],
        daytime.power_or_irradiance(ghi)
    )


def test_get_sunrise_left_alignment(daytime_mask_left_aligned):
    sunrise_left_aligned = daytime.get_sunrise(daytime_mask_left_aligned,
                                               data_alignment='L')
    # Assert that the output time series index is the same as the input
    pd.testing.assert_index_equal(sunrise_left_aligned.index,
                                  daytime_mask_left_aligned.index)
    # Check that the output matches expected
    sunrise_3_19 = sunrise_left_aligned[sunrise_left_aligned.index.date ==
                                        date(2022, 3, 19)]
    # Assert all values for the day equal '2022-03-19 06:10:00-07:00'
    assert all(sunrise_3_19 == pd.to_datetime('2022-03-19 06:10:00-07:00'))


def test_get_sunrise_center_alignment(daytime_mask_center_aligned):
    sunrise_center_aligned = daytime.get_sunrise(daytime_mask_center_aligned,
                                                 data_alignment='C')
    # Assert that the output time series index is the same as the input
    pd.testing.assert_index_equal(sunrise_center_aligned.index,
                                  daytime_mask_center_aligned.index)
    # Check that the output matches expected
    sunrise_3_19 = sunrise_center_aligned[sunrise_center_aligned.index.date ==
                                          date(2022, 3, 19)]
    # Assert all values for the day equal '2022-03-19 06:10:00-07:00'
    assert all(sunrise_3_19 == pd.to_datetime('2022-03-19 06:10:00-07:00'))


def test_get_sunrise_right_alignment(daytime_mask_right_aligned):
    sunrise_right_aligned = daytime.get_sunrise(daytime_mask_right_aligned,
                                                data_alignment='R')
    # Assert that the output time series index is the same as the input
    pd.testing.assert_index_equal(sunrise_right_aligned.index,
                                  daytime_mask_right_aligned.index)
    # Check that the output matches expected
    sunrise_3_19 = sunrise_right_aligned[sunrise_right_aligned.index.date ==
                                         date(2022, 3, 19)]
    # Assert all values for the day equal '2022-03-19 06:10:00-07:00'
    assert all(sunrise_3_19 == pd.to_datetime('2022-03-19 06:10:00-07:00'))


def test_get_sunset_left_alignment(daytime_mask_left_aligned):
    sunset_left_aligned = daytime.get_sunset(daytime_mask_left_aligned,
                                             data_alignment='L')
    # Assert that the output time series index is the same as the input
    pd.testing.assert_index_equal(sunset_left_aligned.index,
                                  daytime_mask_left_aligned.index)
    # Check that the output matches expected
    sunset_3_19 = sunset_left_aligned[sunset_left_aligned.index.date ==
                                      date(2022, 3, 19)]
    # Assert all values for the day equal '2022-03-19 06:10:00-07:00'
    assert all(sunset_3_19 == pd.to_datetime('2022-03-19 17:55:00-07:00'))


def test_get_sunset_center_alignment(daytime_mask_center_aligned):
    sunset_center_aligned = daytime.get_sunset(daytime_mask_center_aligned,
                                               data_alignment='C')
    # Assert that the output time series index is the same as the input
    pd.testing.assert_index_equal(sunset_center_aligned.index,
                                  daytime_mask_center_aligned.index)
    # Check that the output matches expected
    sunset_3_19 = sunset_center_aligned[sunset_center_aligned.index.date ==
                                        date(2022, 3, 19)]
    # Assert all values for the day equal '2022-03-19 06:10:00-07:00'
    assert all(sunset_3_19 == pd.to_datetime('2022-03-19 17:55:00-07:00'))


def test_get_sunset_right_alignment(daytime_mask_right_aligned):
    sunset_right_aligned = daytime.get_sunset(daytime_mask_right_aligned,
                                              data_alignment='R')
    # Assert that the output time series index is the same as the input
    pd.testing.assert_index_equal(sunset_right_aligned.index,
                                  daytime_mask_right_aligned.index)
    # Check that the output matches expected
    sunset_3_19 = sunset_right_aligned[sunset_right_aligned.index.date ==
                                       date(2022, 3, 19)]
    # Assert all values for the day equal '2022-03-19 06:10:00-07:00'
    assert all(sunset_3_19 == pd.to_datetime('2022-03-19 17:55:00-07:00'))


def test_sunrise_alignment_error(daytime_mask_left_aligned):
    with pytest.raises(ValueError,
                       match=("No valid data alignment given. Please pass 'L'"
                              " for left-aligned data, 'R' for "
                              "right-aligned data, or 'C' for "
                              "center-aligned data.")):
        daytime.get_sunrise(daytime_mask_left_aligned,
                            data_alignment='M')


def test_sunset_alignment_error(daytime_mask_left_aligned):
    with pytest.raises(ValueError,
                       match=("No valid data alignment given. Please pass 'L'"
                              " for left-aligned data, 'R' for "
                              "right-aligned data, or 'C' for "
                              "center-aligned data.")):
        daytime.get_sunset(daytime_mask_left_aligned,
                           data_alignment='M')


def test_consistent_modeled_midday_series(daytime_mask_right_aligned,
                                          daytime_mask_left_aligned,
                                          daytime_mask_center_aligned,
                                          modeled_midday_series):
    # Get sunrise and sunset times for each time series (left, right, center),
    # calculate the midday point, compared to modeled midday,
    # and compare that it's consistent across all three series
    # Right-aligned data
    right_sunset = daytime.get_sunset(daytime_mask_right_aligned,
                                      data_alignment='R')
    right_sunrise = daytime.get_sunrise(daytime_mask_right_aligned,
                                        data_alignment='R')
    midday_series_right = right_sunrise + ((right_sunset - right_sunrise)/2)
    midday_series_right.index = midday_series_right.index.date
    midday_diff_right = (modeled_midday_series -
                         midday_series_right.drop_duplicates())
    # Left-aligned data
    left_sunset = daytime.get_sunset(daytime_mask_left_aligned,
                                     data_alignment='L')
    left_sunrise = daytime.get_sunrise(daytime_mask_left_aligned,
                                       data_alignment='L')
    midday_series_left = left_sunrise + ((left_sunset - left_sunrise)/2)
    midday_series_left.index = midday_series_left.index.date
    midday_diff_left = (modeled_midday_series -
                        midday_series_left.drop_duplicates())
    # Center-aligned data
    center_sunset = daytime.get_sunset(daytime_mask_center_aligned,
                                       data_alignment='C')
    center_sunrise = daytime.get_sunrise(daytime_mask_center_aligned,
                                         data_alignment='C')
    midday_series_center = center_sunrise + ((center_sunset -
                                              center_sunrise)/2)
    midday_series_center.index = midday_series_center.index.date
    midday_diff_center = (modeled_midday_series -
                          midday_series_center.drop_duplicates())
    assert (midday_diff_right.equals(midday_diff_left) &
            midday_diff_center.equals(midday_diff_right))
    # Assert that the difference between modeled midday for midday
    # center-aligned data (and consequently left- and right-aligned,
    # which are asserted above as identical to center-aligned data) is less
    # than 10 minutes/600 seconds (this threshold was generally considered
    # noise in the time shift detection paper).
    assert all(midday_diff_center.dt.total_seconds().abs() <= 600)
