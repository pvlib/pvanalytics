"""Tests for data shift quality control functions."""
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
import ruptures as rpt
import pytest
from pandas.util.testing import assert_series_equal
import matplotlib.pyplot as plt
#from pvanalytics.quality import data_shifts as dt
import data_shifts as dt

#@pytest.fixture
def generate_daily_time_series():
    location = pvlib.location.Location(lat, long)
    years = ['2014', '2015', '2016', '2017',
             '2018', '2019']
    # # Add datetime index to series
    # time_range = pd.date_range('2016-01-01T00:00:00.000Z',
    #                            '2018-06-06T00:00:00.000Z', freq='1D')
    # signal, bkps = rpt.pw_wavy(800, 0, noise_std=20)
    # # Add a changepoint in the middle of the signal sequence
    # signal[250:] = signal[250:] + 50
    # # Create pandas series with datetime index and no datetime index
    # signal_no_index = pd.Series(signal)
    # signal_datetime_index = pd.Series(signal)
    # signal_datetime_index.index = pd.to_datetime(time_range[:800])
    # # Create a third time series that is less than 365 days in length to test on
    # signal_datetime_index_short = signal_datetime_index[:300]
    # return signal_no_index, signal_datetime_index, signal_datetime_index_short

def test_detect_data_shifts():
    """
    Unit test that data shifts are correctly identified in the simulated time 
    series.
    """
    signal_no_index, signal_datetime_index, signal_short = generate_daily_time_series()
    # Test that an error is thrown when a Pandas series with no datetime index is
    # passed
    pytest.raises(TypeError, dt.detect_data_shifts, signal_no_index)
    # Test that an error is thrown when an incorrect ruptures method is passed
    pytest.raises(TypeError, dt.detect_data_shifts, signal_datetime_index)
    # Test that an error is thrown when an integer isn't passed in the penalty variable
    pytest.raises(TypeError, dt.detect_data_shifts, signal_datetime_index)
    # Test that an error is thrown when an incorrect string is passed as the cost variable
    pytest.raises(TypeError, dt.detect_data_shifts, signal_datetime_index)
    # Test that a data shift is successfully detecting at index 250 for the datetime-
    # indexed time series
    shift_indices = dt.detect_data_shifts(time_series = signal_datetime_index)
    
    

def test_filter_data_shifts():
    """
    Unit test that the longest interval between data shifts is selected for
    the simulated daily time series data set.
    """
    signal_no_index, signal_datetime_index = generate_daily_time_series()
    