"""Tests for data shift quality control functions."""
import pandas as pd
import ruptures as rpt
import pytest
from pvanalytics.quality import data_shifts as dt


#@pytest.fixture
def generate_daily_time_series():
    # Pull down the saved PVLib dataframe and process it
    df = pd.read_csv("https://datahub.duramat.org/dataset/7b72ae24-c0c2-4339"
                     "-93dd-2c9c10d64c90/resource/a2f73100-2482-4d9f-a348-"
                     "c45a6512964f/download/"
                     "pvlib_data_shift_stream_example_1.csv")
    df = df.iloc[1:, :]
    signal_no_index = df['value']
    df.index = pd.to_datetime(df['timestamp'])
    signal_datetime_index = df['value']
    changepoint_date = df[df['label'] == 1].index[0]
    return signal_no_index, signal_datetime_index, changepoint_date


def test_detect_data_shifts(generate_daily_time_series):
    """
    Unit test that data shifts are correctly identified in the simulated time
    series.
    """
    signal_no_index, signal_datetime_index, changepoint_date = \
        generate_daily_time_series()
    # Test that an error is thrown when a Pandas series with no datetime
    # index is passed
    pytest.raises(TypeError, dt.detect_data_shifts, signal_no_index)
    # Test that an error is thrown when an incorrect ruptures method is passed
    pytest.raises(TypeError, dt.detect_data_shifts, signal_datetime_index,
                  True, False, "Pelt")
    pytest.raises(TypeError, dt.detect_data_shifts, signal_datetime_index,
                  True, False, rpt.Dynp)
    # Test that an error is thrown when an incorrect string is passed as the
    # cost variable
    pytest.raises(TypeError, dt.detect_data_shifts, signal_datetime_index,
                  True, False, rpt.Binseg, "none")
    # Test that an error is thrown when an integer isn't passed in the
    # penalty variable
    pytest.raises(TypeError, dt.detect_data_shifts, signal_datetime_index,
                  True, False, rpt.Binseg, "rbf", 3.14)
    # Test that a warning is thrown when the data is less than 2 years
    # in length
    pytest.warns(UserWarning, dt.detect_data_shifts,
                 signal_datetime_index[:500], True, True)
    # Test that a data shift is successfully detected within 5 days of
    # inserted changepoint
    shift_index = dt.detect_data_shifts(time_series=signal_datetime_index)
    # Test that the column name is handled if a series with no name is passed
    signal_unnamed = signal_datetime_index.rename(None)
    shift_index_unnamed = dt.detect_data_shifts(signal_unnamed)
    # Run model with manually entered parameters
    shift_index_param = dt.detect_data_shifts(signal_datetime_index, True,
                                              False)
    assert (abs((changepoint_date - shift_index[0]).days) <= 5) & \
        (abs((changepoint_date - shift_index_unnamed[0]).days) <= 5) &\
        (abs((changepoint_date - shift_index_param[0]).days) <= 5)


def test_filter_data_shifts(generate_daily_time_series):
    """
    Unit test that the longest interval between data shifts is selected for
    the simulated daily time series data set.
    """
    signal_no_index, signal_datetime_index, changepoint_date = \
        generate_daily_time_series
    # Run the time series where there are no changepoints
    interval_dict_short = dt.filter_data_shifts(
        time_series=signal_datetime_index[:100])
    # Run the time series where there is a changepoint
    interval_dict = dt.filter_data_shifts(
        time_series=signal_datetime_index)
    assert (interval_dict['start_date'] == pd.to_datetime('2015-10-30')) & \
        (interval_dict['end_date'] == pd.to_datetime('2020-12-31'))
    assert (interval_dict_short['start_date'] ==
            signal_datetime_index.index.min()) & \
        (interval_dict_short['end_date'] ==
         signal_datetime_index[:100].index.max())
