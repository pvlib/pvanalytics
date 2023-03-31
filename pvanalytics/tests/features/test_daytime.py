"""Tests for :py:mod:`features.daytime`"""
import pytest
import pandas as pd
import numpy as np
from pvlib.location import Location
from pvanalytics.features import daytime


@pytest.fixture(scope='module',
                params=['H', '15T', pytest.param('T', marks=pytest.mark.slow)])
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


def _assert_daytime_no_shoulder(clearsky, output):
    # every night-time value in `output` has low or 0 irradiance
    assert all(clearsky[~output] < 3)
    if pd.infer_freq(clearsky.index) == 'T':
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
    clearsky = location.get_clearsky(
        pd.date_range(start='1/1/2020', end='1/10/2020', freq='15T'),  # no tz
        model='simplified_solis'
    )
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
        freq='15T',
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
        freq='15T',
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
