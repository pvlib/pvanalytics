"""Tests for utility functions."""
import pandas as pd
from pandas.util.testing import assert_series_equal
import pytest
from pvanalytics.quality import util


def test_check_limits():
    """Test the private check limits function.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    expected = pd.Series(data=[True, False])
    data = pd.Series(data=[3, 2])
    result = util.check_limits(val=data, lower_bound=2.5)
    assert_series_equal(expected, result, check_names=False)
    result = util.check_limits(val=data, lower_bound=3, inclusive_lower=True)
    assert_series_equal(expected, result, check_names=False)

    data = pd.Series(data=[3, 4])
    result = util.check_limits(val=data, upper_bound=3.5)
    assert_series_equal(expected, result, check_names=False)
    result = util.check_limits(val=data, upper_bound=3, inclusive_upper=True)
    assert_series_equal(expected, result, check_names=False)

    result = util.check_limits(
        val=data,
        lower_bound=3,
        upper_bound=4,
        inclusive_lower=True,
        inclusive_upper=True
    )
    assert all(result)
    result = util.check_limits(val=data, lower_bound=3, upper_bound=4)
    assert not any(result)

    with pytest.raises(ValueError):
        util.check_limits(val=data)


@pytest.fixture
def ten_days():
    """A ten day index (not localized) at ten minute frequency."""
    return pd.date_range(
        start='01/01/2020',
        end='01/10/2020 23:59',
        freq='10T'
    )


def test_daily_min_all_below(ten_days):
    """If every data point is below the minimum then all values are
    flagged False."""
    data = pd.Series(1, index=ten_days)
    assert_series_equal(
        pd.Series(False, index=ten_days),
        util.daily_min(data, minimum=2)
    )


def test_daily_min_some_values_above(ten_days):
    """If there are some values above `maximum` on one day, all data is
    flagged False."""
    data = pd.Series(1, index=ten_days)
    data.iloc[0:10] = 10
    data.iloc[100:150] = 10
    data.iloc[500:710] = 3
    assert_series_equal(
        pd.Series(False, index=ten_days),
        util.daily_min(data, minimum=2)
    )


def test_daily_min_one_day_fails(ten_days):
    """daily_maxmin flags day with all values above maximum True."""
    data = pd.Series(2, index=ten_days)
    data.loc['01/02/2020'] = 5
    expected = pd.Series(False, index=ten_days)
    expected['01/02/2020'] = True
    assert_series_equal(
        expected,
        util.daily_min(data, minimum=3)
    )


def test_daily_min_exclusice(ten_days):
    """Check that inclusive=False makes the comparison greater than or
    equal."""
    data = pd.Series(2, index=ten_days)
    assert not util.daily_min(data, minimum=2, inclusive=False).any()
    assert util.daily_min(data, minimum=2, inclusive=True).all()
