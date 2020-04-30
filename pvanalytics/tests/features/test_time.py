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


def test_daytime_only(quadratic_day):
    """In a series with only daytime data, all points are flagged."""
    daylight = time.daytime(quadratic_day, minimum_days=1)
    assert daylight.iloc[0:61].all()
    assert not daylight.iloc[61:].any()
