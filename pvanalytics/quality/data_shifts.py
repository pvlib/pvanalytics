"""
Quality tests related to detecting and filtering out data shifts in data
streams.
"""

from pvanalytics.quality import gaps
import numpy as np
import pandas as pd
import warnings


def _run_data_checks(time_series):
    """
    Check that the passed parameters can be run through the function.
    This includes checking the passed time series to ensure it has a
    datetime index, and resampling to daily summed data, if needed.

    Parameters
    ----------
    time_series : Pandas series with datetime index.
        Daily time series of a PV data stream, which can include irradiance
        and power data streams. This series represents the summed daily
        values of the particular data stream.

    Returns
    -------
    None.
    """
    # Check that the time series has a datetime index, and consists of numeric
    # values
    if not isinstance(time_series.index, pd.DatetimeIndex):
        raise TypeError('Must be a Pandas series with a datetime index.')
    # Check that the time series is sampled on a daily basis
    if pd.infer_freq(time_series.index) != "D":
        warnings.warn("Time series frequency not set. Setting frequency to "
                      "daily, and resampling the daily sum value.")
        time_series = time_series.resample('d').sum()
    return


def _erroneous_filter(time_series):
    """
    Remove any outliers from the time series.

    Parameters
    ----------
    time_series : Pandas series with datetime index.
        Daily time series of a PV data stream, which can include irradiance
        and power data streams. This series represents the summed daily values
        of the particular data stream.

    Returns
    -------
    time_series: Pandas series, with a datetime index.
        Time series, after filtering out outliers. This includes removal
        of stale repeat readings, negative readings, and data greater than
        the 99th percentile or less than the 1st percentile.
    """
    # Detect and mask stale data
    stale_mask = gaps.stale_values_round(time_series, window=6,
                                         decimals=3, mark='tail')
    # Mask negative and 0 values
    negative_mask = (time_series <= 0)
    # Mask the top 1% and bottom 1% of data points
    quantile_mask = ((time_series <= time_series.quantile(.01)) |
                     (time_series >= time_series.quantile(.99)))
    # Filter out the associated data by masking
    time_series = time_series[(~stale_mask) & (~negative_mask) &
                              (~quantile_mask)]
    return time_series


def _preprocess_data(time_series, remove_seasonality):
    """
    Pre-process the time series, including the following:
        1. Min-max normalization of the time series.
        2. Removing seasonality from the time series.

    Parameters
    ----------
    time_series : Pandas series with datetime index.
        Daily time series of a PV data stream, which can include irradiance
        and power data streams. This series represents the summed daily values
        of the particular data stream.
    remove_seasonality: Boolean.
        Whether or not to remove seasonality from the time series. If set to
        True, the seasonality-removal routine is run on the min-max normalized
        data. If not, the routine is skipped.

    Returns
    -------
    Pandas series with a datetime index:
        Time series, after data processing. This includes min-max
        normalization, and, if the time series is in greater than 2 years in
        length, seasonality removal.
    """
    # Min-max normalize the series
    time_series_normalized = (time_series - time_series.min()) \
        / (time_series.max() - time_series.min())
    # Check if the time series is greater than one year in length. If not, flag
    # a warning and pass back the normalized time series
    if not remove_seasonality:
        return time_series_normalized
    else:
        # Take the median of every day of the year across all years in the
        # data, and use this as the seasonality of the time series
        day_year_values = pd.DatetimeIndex(
            pd.Series(time_series.index)).dayofyear
        time_series_seasonality = time_series_normalized.groupby(
            [day_year_values]).transform("median")
        # Remove seasonlity from the time series
        return (time_series_normalized - time_series_seasonality)


def detect_data_shifts(time_series,
                       filtering=True, use_default_models=True,
                       method=None, cost=None, penalty=40):
    """
    Detect data shifts in the time series, and return list of dates where these
    data shifts occur.

    Parameters
    ----------
    time_series : Pandas series with datetime index.
        Daily time series of a PV data stream, which can include irradiance
        and power data streams. This series represents the summed daily values
        of the particular data stream.
    filtering : Boolean, default True.
        Whether or not to filter out outliers and stale data from the time
        series. If True, then this data is filtered out before running the
        data shift detection sequence. If False, this data is not filtered
        out. Default set to True.
    use_default_models: Boolean, default True
        If True, then default change point detection search parameters are
        used. For time series shorter than 2 years in length, the search
        function is `rpt.Window`  with `model='rbf'`, `width=40` and
        `penalty=30`. For time series 2 years or longer in length, the
        search function is `rpt.BottomUp` with `model='rbf'`
        and `penalty=40`.
    method: ruptures search method instance or None, default None.
        Ruptures search method instance. See
        https://centre-borelli.github.io/ruptures-docs/user-guide/.
    cost: str or None, default None
        Cost function passed to the ruptures changepoint search instance.
        See https://centre-borelli.github.io/ruptures-docs/user-guide/
    penalty: int, default 40
        Penalty value passed to the ruptures changepoint detection method.
        Default set to 40.

    Returns
    -------
    Pandas Series
        Series of boolean values with a datetime index, where detected
        changepoints are labeled as True, and all other values are labeled
        as False.

    References
    -------
    .. [1] Perry K., and Muller, M. "Automated shift detection in sensor-based
       PV power and irradiance time series", 2022 IEEE 48th Photovoltaic
       Specialists Conference (PVSC). Submitted.
    """
    try:
        import ruptures as rpt
    except ImportError:
        raise ImportError("time.shifts_ruptures() requires ruptures.")
    # Run data checks on cleaned data to make sure that the data can be run
    # successfully through the routine
    _run_data_checks(time_series)
    # Run the filtering sequence, if marked as True
    if filtering:
        time_series = _erroneous_filter(time_series)
    # Drop any duplicated data from the time series
    time_series = time_series.drop_duplicates()
    # Check if the time series is more than 2 years long. If so, remove
    # seasonality. If not, run analysis on the normalized time series
    if (time_series.index.max() - time_series.index.min()).days <= 730:
        warnings.warn("The passed time series is less than 2 years in length, "
                      "and will not be corrected for seasonality. Runnning "
                      "data shift detection on the min-max normalized time "
                      "series with no seasonality correction.")
        time_series_processed = _preprocess_data(time_series,
                                                 remove_seasonality=False)
        seasonality_rmv = False
    else:
        # Perform pre-processing on the time series, to get the
        # seasonality-removed time series.
        time_series_processed = _preprocess_data(time_series,
                                                 remove_seasonality=True)
        seasonality_rmv = True
    points = np.array(time_series_processed.dropna())
    # If seasonality has been removed and default model is used, run
    # BottomUp method
    if (seasonality_rmv) & (use_default_models):
        algo = rpt.BottomUp(model='rbf').fit(points)
        result = algo.predict(pen=40)
    # If there is no seasonality but default model is used, run
    # Window-based method
    elif (not seasonality_rmv) & (use_default_models):
        algo = rpt.Window(model='rbf',
                          width=50).fit(points)
        result = algo.predict(pen=30)
    # Otherwise run changepoint detection with the passed parameters
    else:
        algo = method(model=cost).fit(points)
        result = algo.predict(pen=penalty)
    # Remove the last index of the time series, if present
    if len(points) in result:
        result.remove(len(points))
    time_series_processed.index.name = "datetime"
    mask = pd.Series(False, index=time_series_processed.index)
    mask.iloc[result] = True
    return mask


def get_longest_shift_segment_dates(time_series,
                                    filtering=True,
                                    use_default_models=True,
                                    method=None, cost=None,
                                    penalty=40):
    """
    Return the start and end dates of the longest continuous time series
    segment. During this process, data shift detection is performed, and the
    longest time series segment between changepoints is identified, and the
    start and end dates of that segment are returned.

    Parameters
    ----------
    time_series : Pandas series with datetime index.
        Daily time series of a PV data stream, which can include irradiance
        and power data streams. This series represents the summed daily values
        of the particular data stream.
    filtering : Boolean, default True.
        Whether or not to filter out outliers and stale data from the time
        series. If True, then this data is filtered out before running the
        data shift detection sequence. If False, this data is not filtered
        out. Default set to True.
    use_default_models: Boolean, default True
        If True, then default change point detection search parameters are
        used. For time series shorter than 2 years in length, the search
        function is `rpt.Window`  with `model='rbf'`, `width=40` and
        `penalty=30`. For time series 2 years or longer in length, the
        search function is `rpt.BottomUp` with `model='rbf'`
        and `penalty=40`.
    method: ruptures search method instance or None, default None.
        Ruptures search method instance. See
        https://centre-borelli.github.io/ruptures-docs/user-guide/.
    cost: str or None, default None
        Cost function passed to the ruptures changepoint search instance.
        See https://centre-borelli.github.io/ruptures-docs/user-guide/
    penalty: int, default 40
        Penalty value passed to the ruptures changepoint detection method.
        Default set to 40.

    Returns
    -------
    passing_dates_dict: Dictionary.
        Dictionary object containing the longest continuous time segment
        with no detected data shifts. The start date of the period is
        represented in the "start_date" field, and the end date of the
        period is represented in the "end_date" field.

    References
    -------
    .. [1] Perry K., and Muller, M. "Automated shift detection in sensor-based
       PV power and irradiance time series", 2022 IEEE 48th Photovoltaic
       Specialists Conference (PVSC). Submitted.
    """
    # Detect indices where data shifts occur
    cpd_mask = detect_data_shifts(time_series, filtering,
                                  use_default_models,
                                  method, cost, penalty)
    interval_id = cpd_mask.cumsum()
    longest_interval_id = interval_id.value_counts().idxmax()
    index = interval_id.index[interval_id == longest_interval_id]
    passing_dates_dict = {'start_date': index.min(), 'end_date': index.max()}
    return passing_dates_dict
