"""Tests for time-related quality control functions."""
from datetime import datetime
import pytz
import pytest
import pandas as pd
from pandas.util.testing import assert_series_equal
from pvanalytics.quality import time


@pytest.fixture
def times():
    """One hour in Mountain Standard Time at 10 minute intervals.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    MST = pytz.timezone('MST')
    return pd.date_range(start=datetime(2018, 6, 15, 12, 0, 0, tzinfo=MST),
                         end=datetime(2018, 6, 15, 13, 0, 0, tzinfo=MST),
                         freq='10T')


def test_timestamp_spacing_date_range(times):
    """An index generated by pd.date_range has the expected spacing."""
    assert_series_equal(
        time.spacing(times, times.freq),
        pd.Series(True, index=times)
    )


def test_timestamp_spacing_one_timestamp(times):
    """An index with only one timestamp has uniform spacing."""
    assert_series_equal(
        time.spacing(times[[0]], times.freq),
        pd.Series(True, index=[times[0]])
    )


def test_timestamp_spacing_one_missing(times):
    """The timestamp following a missing timestamp will be marked False."""
    assert_series_equal(
        time.spacing(times[[0, 2, 3]], times.freq),
        pd.Series([True, False, True], index=times[[0, 2, 3]])
    )


def test_timestamp_spacing_too_frequent(times):
    """Timestamps with too high frequency will be marked False."""
    assert_series_equal(
        time.spacing(times, '30min'),
        pd.Series([True] + [False] * (len(times) - 1), index=times)
    )


@pytest.mark.parametrize("tz", ['MST', 'America/Denver'])
def test_has_dst(tz, albuquerque):
    days = pd.date_range(
        start='1/1/2020',
        end='1/1/2021',
        freq='D',
        tz=tz
    )
    sunrise = albuquerque.get_sun_rise_set_transit(
        days, method='spa')['sunrise']
    dst = time.has_dst(sunrise, ['3/8/2020', '11/1/2020'])
    if tz == 'America/Denver':
        assert dst == [True, True]
    else:
        assert dst == [False, False]
