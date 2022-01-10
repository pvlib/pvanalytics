"""Tests for data shift quality control functions."""
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
import ruptures as rpt
import pytest
from pandas.util.testing import assert_series_equal

from pvanalytics.quality import data_shifts


@pytest.fixture
def generate_daily_time_series():
    power_datetime_index = pd.Series(np.arange(1, 51))
    power_datetime_index = pd.concat([power_datetime_index,
                                      power_datetime_index[::-1]])
    # Add datetime index to second series
    time_range = pd.date_range('2016-12-02T11:00:00.000Z',
                               '2017-06-06T07:00:00.000Z', freq='1D')
    power_datetime_index.index = pd.to_datetime(time_range[:100])
    # Note: Power is expected to be Series object with a datetime index.
    return power_datetime_index

def test_detect_data_shifts():
    """
    Test that data shifts are correctly identified in the simulated time 
    series.
    """
    pass

def test_filter_data_shifts():
    """
    Test that the longest interval between data shifts is selected for
    the simulated daily time series data set.
    """
    pass