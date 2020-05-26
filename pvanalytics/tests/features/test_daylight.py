"""Tests for functions in features.daylight"""
import pytest
import pandas as pd
from pvanalytics.features import daylight
from pandas.util.testing import assert_series_equal


def test_frequency_too_few_days(quadratic):
    """daylight.frequency raises an error when there is not enough
    data."""
    quadratic.index = pd.date_range(
        start='01/01/2020 07:30', freq='10T', periods=len(quadratic)
    )
    expected_exception = ("Too few days with data "
                          "[(]got [0-9]+, minimum_days=[0-9]+[)]")
    with pytest.raises(ValueError, match=expected_exception):
        daylight.frequency(quadratic, minimum_days=2)


def test_frequency_one_day_only(quadratic_day):
    """In a series with one quadratic only the quadratic is flagged as
    daytime"""
    daylight_times = daylight.frequency(quadratic_day, minimum_days=1)
    # the roots of the quadratic (at 0 & 61) are 0 so will be excluded
    # from the mask returned by daylight.frequency()
    assert daylight_times.iloc[1:60].all()
    assert not daylight_times.iloc[60:].any()


def test_frequency_only(quadratic_day):
    """If every value is > 0 then all values are flagged True."""
    assert daylight.frequency(
        quadratic_day.iloc[1:60], minimum_days=0).all()


def test_frequency_zero_only(quadratic_day):
    """If evey value == 0 then all values are flagged False."""
    assert not daylight.frequency(
        quadratic_day.iloc[60:], minimum_days=0).any()


def test_level_one_day(quadratic_day):
    """level identifies the single period of positive."""
    assert not daylight.level(quadratic_day).iloc[61:0].any()
    assert (quadratic_day[daylight.level(quadratic_day)] > 0).all()


def test_level_threshold_zero(quadratic_day):
    """If threshold=0.0 then every datapoint is marked True."""
    assert daylight.level(quadratic_day, threshold=0).all()


def test_level_nonzero_night(quadratic_day):
    """Low, but nonzero, nightime values are excluede."""
    quadratic_day.iloc[61:] = 10
    assert not daylight.level(quadratic_day).iloc[61:0].any()
    assert (quadratic_day[daylight.level(quadratic_day)] > 10).all()


def test_level_daytime_split(quadratic_day):
    """The correct mask is returned when daytime spans two dates"""
    shifted_day = quadratic_day.shift(freq='20H')
    assert_series_equal(
        daylight.level(quadratic_day).shift(freq='20H'),
        daylight.level(shifted_day)
    )


def test_frequency_daytime_split(quadratic_day):
    """The correct mask is returned when daytime spans two dates"""
    shifted_day = quadratic_day.shift(freq='20H')
    assert_series_equal(
        daylight.frequency(quadratic_day, minimum_days=1).shift(freq='20H'),
        daylight.frequency(shifted_day, minimum_days=1)
    )
