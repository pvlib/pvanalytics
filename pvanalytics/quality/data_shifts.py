"""
Quality tests related to detecting and filtering out data shifts in data streams.
"""

from pvanalytics.quality import gaps
import numpy as np
import pandas as pd
import ruptures as rpt
import warnings
import abc


def _run_data_checks(time_series, method, cost, penalty):
    """
    Check that the passed parameters can be run through the function. 
    This includes checking the passed time series, method, cost, 
    and penalty values.

    Parameters
    ----------
    time_series : Pandas series with datetime index.
        DESCRIPTION.
    method: ruptures search method object.
    cost: str
    penalty: int

    Returns
    -------
    None.
    """
    # Check that the time series has a datetime index, and consists of numeric
    # values
    if not isinstance(time_series.index, pd.DatetimeIndex):
        raise TypeError('Must be a Pandas series with a datetime index.')
    # Check that the method passed is one of the approved ruptures methods
    if type(method) !=  abc.ABCMeta:
        raise TypeError("Method must be of type: ruptures.Pelt, "\
                         "ruptures.Binseg, ruptures.BottomUp, or ruptures.Window.")         
    if (method.__name__ != "Pelt") & \
        (method.__name__ != "Binseg") &\
        (method.__name__ != "BottomUp") &\
        (method.__name__ != "Window"):
        raise TypeError("Method must be of type: ruptures.Pelt, "\
                         "ruptures.Binseg, ruptures.BottomUp, or ruptures.Window.")    
    # Check that the cost passed is one of the approved ruptures costs
    if (cost != "rbf") & (cost != "l1") & (cost != "l2") & (cost != "normal") &\
        (cost != "cosine") & (cost != "linear"):
        raise TypeError("Cost must be of type: 'rbf', 'l1', 'l2', 'normal', "\
                         "'cosine', or 'linear'.")
    # Check that the penalty is an int value
    if not isinstance(penalty, int):
        raise TypeError('Penalty value must be an integer.')
    return


def _erroneous_filter(time_series): 
    """
    Remove any outliers from the time series.
    
    Parameters
    ----------
    time_series : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    """
    time_series = time_series.copy()
    # Detect and mask stale data 
    stale_mask = gaps.stale_values_round(time_series, window=6, 
                                         decimals=3, mark='tail')
    time_series = time_series[~stale_mask]
    # Set in negative values to NaN
    time_series.loc[time_series <= 0] = np.nan
    #Remove the top 10% and bottom 10% data points, and keep everything else intact
    time_series[(time_series <= time_series.quantile(.01)) |
                (time_series >= time_series.quantile(.99))] = np.nan
    time_series = time_series.drop_duplicates()
    return time_series


def _preprocess_data(time_series):
    """
    Pre-process the time series, including the following:
        1. Min-max normalization of the time series.
        2. Removing seasonality from the time series. 

    Parameters
    ----------
    time_series : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    """
    # Convert the time series to a dataframe to do the pre-processing
    column_name = time_series.name
    if column_name is None:
        column_name = "value"
        time_series = time_series.rename(column_name)
    df = time_series.to_frame()
    # Min-max normalize the series
    df[column_name + "_normalized"] = (df[column_name] - df[column_name].min())/(df[column_name].max() - df[column_name].min())
    # Check if the time series is greater than one year in length. If not, flag a warning
    # and pass back the normalized time series
    if (df.index.max() - df.index.min()).days <= 730:
        raise Warning("The passed time series is less than 2 years in length, and "
                      "cannot be corrected for seasonality. Runnning data shift detection "
                      "on the min-max normalized time series (NO seasonality correction).")
        return df[column_name + "_normalized"]
    else:
        # Take the average of every day of the year across all years in the data set and use this
        # as the seasonality of the time series
        df['month'] = pd.DatetimeIndex(df.index).month
        df['day'] = pd.DatetimeIndex(pd.Series(df.index)).day
        df['seasonal_val'] = df.groupby(['month','day'])[column_name + "_normalized"].transform("median")
        return df[column_name + "_normalized"] - df['seasonal_val']


def detect_data_shifts(time_series, filtering=True, method = rpt.BottomUp,
                       cost = "rbf", penalty = 40):
    """
    Detect data shifts in the time series, and return list of dates where these data
    shifts occur.
    
    Parameters
    ----------
    time_series : TYPE
        DESCRIPTION.
    filtering : Boolean.
    method: ruptures search method object.
    cost: str
    penalty: int

    Returns
    -------
    None.
    """
    # Run data checks on cleaned data to make sure that the data can be run successfully
    # through the routine
    _run_data_checks(time_series, method, cost, penalty)
    # Run the filtering sequence, if marked as True
    if filtering:
        time_series = _erroneous_filter(time_series)
    # Perform pre-processing on the time series, to get the seasonality-removed
    # time series.
    time_series_processed = _preprocess_data(time_series)  
    # Run changepoint detection on the time series
    algo = method(model=cost).fit(np.array(time_series_processed))
    result = algo.predict(pen=penalty)
    # Remove the first and last indices of the time series, if present
    if len(time_series) in result:
        result.remove(len(time_series))
    if 1 in result:
        result.remove(1)
    # Return a list of dates where changepoints are detected
    time_series_processed.index.name = "datetime" 
    time_series_processed = time_series_processed.reset_index()
    return list(time_series_processed.loc[result]['datetime'])
        

def filter_data_shifts(time_series, filtering=True,
                       method = rpt.BottomUp, cost = "rbf", penalty = 40):
    """
    Filter the time series by the longest continuous time series segment, by performing
    data shift detection.
    
    Parameters
    ----------
    time_series : TYPE
        DESCRIPTION.
    filtering : Boolean.
    method: ruptures search method object.
    cost: str
    penalty: int

    Returns
    -------
    None.
    """    
    # Detect indices where data shifts occur
    data_shift_dates = detect_data_shifts(time_series, filtering,
                                          method, cost, penalty)
    # Get longest continuous data segment, by number of days in the time series.
    if not data_shift_dates:
        print("No data shifts detected in the time series. Returning the full time series dates...")
        passing_dates_dict = {"start_date": time_series.index.min(),
                              "end_date": time_series.index.max()
                                }
        return passing_dates_dict
    else:
        # Find the longest date segment in the time series, with the most data points.
        segment_lengths = [len(time_series[data_shift_dates[i]:data_shift_dates[i+1]]) \
                           for i in range(len(data_shift_dates)-1)]
        max_segment_length = segment_lengths.index(max(segment_lengths))
        passing_dates_dict = {"start_date": data_shift_dates[max_segment_length-1],
                              "end_date": data_shift_dates[max_segment_length]
                              }
        return passing_dates_dict
        
        pass
    
    
    