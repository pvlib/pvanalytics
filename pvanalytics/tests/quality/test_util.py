"""Tests for utility functions."""
import pandas as pd
from pandas.util.testing import assert_series_equal
import pytest
from pvanalytics.quality import util


def test_check_limits():
    """Test the private check limits function."""
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

    result = util.check_limits(val=data, lower_bound=3, upper_bound=4, inclusive_lower=True,
                                      inclusive_upper=True)
    assert all(result)
    result = util.check_limits(val=data, lower_bound=3, upper_bound=4)
    assert not any(result)

    with pytest.raises(ValueError):
        util.check_limits(val=data)
