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
    # Pull down the saved PVLib dataframe and process it
    df = pd.read_csv("C:/Users/kperry/Documents/source/repos/pvanalytics/pvanalytics/data/pvlib_data_shift_data_stream.csv")
    signal_no_index = df['value']
    df.index = pd.to_datetime(df['timestamp'])
    signal_datetime_index = df['value']
    changepoint_date = df[df['label'] == 1].index[0]
    return signal_no_index, signal_datetime_index, changepoint_date

def test_detect_data_shifts():
    """
    Unit test that data shifts are correctly identified in the simulated time 
    series.
    """
    signal_no_index, signal_datetime_index, changepoint_date = generate_daily_time_series()
    # Test that an error is thrown when a Pandas series with no datetime index is
    # passed
    pytest.raises(TypeError, dt.detect_data_shifts, signal_no_index)
    # Test that an error is thrown when an incorrect ruptures method is passed
    pytest.raises(TypeError, dt.detect_data_shifts, signal_datetime_index, True, "Pelt")
    pytest.raises(TypeError, dt.detect_data_shifts, signal_datetime_index, True, rpt.Dynp)
    # Test that an error is thrown when an incorrect string is passed as the cost variable
    pytest.raises(TypeError, dt.detect_data_shifts, signal_datetime_index, True, rpt.Binseg,
                  "none")
    # Test that an error is thrown when an integer isn't passed in the penalty variable
    pytest.raises(TypeError, dt.detect_data_shifts, signal_datetime_index, True, rpt.Binseg,
                  "rbf", 3.14)
    # Test that a warning is thrown when the time series is less than 2 years long.
    pytest.warns(UserWarning, dt.detect_data_shifts, signal_datetime_index[:500])
    # Test that a data shift is successfully detecting at index 250 for the datetime-
    # indexed time series
    shift_indices = dt.detect_data_shifts(time_series = signal_datetime_index)
    assert shift_indices == []

def test_filter_data_shifts():
    """
    Unit test that the longest interval between data shifts is selected for
    the simulated daily time series data set.
    """
    signal_no_index, signal_datetime_index, changepoint_date = generate_daily_time_series()
    # Run the time series where there are no changepoints
    dt.filter_data_shifts(time_series = signal_datetime_index)
    # Run the time series where there is a changepoint
    dt.filter_data_shifts(time_series = signal_datetime_index)
    
    
    