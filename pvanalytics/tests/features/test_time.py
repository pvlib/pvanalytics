"""Tests for functions in features.time."""
import pytest
import pandas as pd
from pvanalytics.features import time


def test_daytime_frequency_too_few_days(quadratic):
    """time.daytime_frequency raises an error when there is not enough
    data."""
    quadratic.index = pd.date_range(
        start='01/01/2020 07:30', freq='10T', periods=len(quadratic)
    )
    expected_exception = ("Too few days with data "
                          "[(]got [0-9]+, minimum_days=[0-9]+[)]")
    with pytest.raises(ValueError, match=expected_exception):
        time.daytime_frequency(quadratic, minimum_days=2)


def test_daytime_frequency_one_day_only(quadratic_day):
    """In a series with one quadratic only the quadratic is flagged as
    daytime"""
    daylight = time.daytime_frequency(quadratic_day, minimum_days=1)
    # the roots of the quadratic (at 0 & 61) are 0 so will be excluded
    # from the mask returned by time.daytime_frequency()
    assert daylight.iloc[1:60].all()
    assert not daylight.iloc[60:].any()


def test_daytime_frequency_only(quadratic_day):
    """If every value is > 0 then all values are flagged True."""
    assert time.daytime_frequency(
        quadratic_day.iloc[1:60], minimum_days=0).all()


def test_daytime_frequency_zero_only(quadratic_day):
    """If evey value == 0 then all values are flagged False."""
    assert not time.daytime_frequency(
        quadratic_day.iloc[60:], minimum_days=0).any()


def test_daytime_level_one_day(quadratic_day):
    """daytime_level identifies the single period of positive."""
    assert not time.daytime_level(quadratic_day).iloc[61:0].any()
    assert (quadratic_day[time.daytime_level(quadratic_day)] > 0).all()


def test_daytime_level_threshold_zero(quadratic_day):
    """If threshold=0.0 then every datapoint is marked True."""
    assert time.daytime_level(quadratic_day, threshold=0).all()


def test_daytime_level_nonzero_night(quadratic_day):
    """Low, but nonzero, nightime values are excluede."""
    quadratic_day.iloc[61:] = 10
    assert not time.daytime_level(quadratic_day).iloc[61:0].any()
    assert (quadratic_day[time.daytime_level(quadratic_day)] > 10).all()
