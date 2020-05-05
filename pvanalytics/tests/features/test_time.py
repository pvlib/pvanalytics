"""Tests for functions in features.time."""
import pytest
import pandas as pd
from pvanalytics.features import time


def test_daytime_too_few_days(quadratic):
    """time.daytime raises an error when there is not enough data."""
    quadratic.index = pd.date_range(
        start='01/01/2020 07:30', freq='10T', periods=len(quadratic)
    )
    expected_exception = ("Too few days with data "
                          "[(]got [0-9]+, minimum_days=[0-9]+[)]")
    with pytest.raises(ValueError, match=expected_exception):
        time.daytime(quadratic, minimum_days=2)


def test_daytime_one_day_only(quadratic_day):
    """In a series with one quadratic only the quadratic is flagged as
    daytime"""
    daylight = time.daytime(quadratic_day, minimum_days=1)
    # the roots of the quadratic (at 0 & 61) are 0 so will be excluded
    # from the mask returned by time.daytime()
    assert daylight.iloc[1:60].all()
    assert not daylight.iloc[60:].any()


def test_daytime_only(quadratic_day):
    """If every value is > 0 then all values are flagged True."""
    assert time.daytime(quadratic_day.iloc[1:60], minimum_days=0).all()


def test_daytime_zero_only(quadratic_day):
    """If evey value == 0 then all values are flagged False."""
    assert not time.daytime(quadratic_day.iloc[60:], minimum_days=0).any()
