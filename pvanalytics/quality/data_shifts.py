"""
Quality tests related to detecting shifts in data streams.
"""

from pvanalytics.quality import gaps
import numpy as np
import pandas as pd


def _run_data_checks(series):
    """
    Check that the passed parameters can be run through the function.
    This includes checking the passed time series to ensure it has a
    datetime index, and that it is a daily sampled time series.

    Parameters
    ----------
    series : Pandas series with datetime index.
        Daily time series of a PV data stream, which can include irradiance
        and power data streams. This series represents the summed daily
        values of the particular data stream.

    Returns
    -------
    None.
    """
    # Check that the time series has a datetime index, and consists of numeric
    # values
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError('Must be a Pandas series with a datetime index.')
    # Check that the time series is sampled on a daily basis. If not,
    # throw a ValueError exception
    if series.index.to_series().diff().value_counts().idxmax().days != 1:
        raise ValueError("Time series frequency not daily. Please resample "
                         "time series to daily summed values.")
    return


def _erroneous_filter(series):
    """
    Remove any outliers from the time series.

    Parameters
    ----------
    series : Pandas series with datetime index.
        Daily time series of a PV data stream, which can include irradiance
        and power data streams. This series represents the summed daily values
        of the particular data stream.

    Returns
    -------
    series: Pandas series, with a datetime index.
        Time series, after filtering out outliers. This includes removal
        of stale repeat readings, negative readings, and data greater than
        the 99th percentile or less than the 1st percentile.
    """
    # Detect and mask stale data
    stale_mask = gaps.stale_values_round(series, window=6,
                                         decimals=3, mark='tail')
    # Mask negative and 0 values
    negative_mask = (series <= 0)
    # Mask the top 1% and bottom 1% of data points
    quantile_mask = ((series <= series.quantile(.01)) |
                     (series >= series.quantile(.99)))
    # Filter out the associated data by masking
    series = series[(~stale_mask) & (~negative_mask) & (~quantile_mask)]
    return series


def _preprocess_data(series, remove_seasonality):
    """
    Pre-process the time series, including the following:
        1. Min-max normalization of the time series.
        2. Removing seasonality from the time series, if remove_seasonality
        is True.

    Parameters
    ----------
    series : Pandas series with datetime index.
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
    series_normalized = (series - series.min()) \
        / (series.max() - series.min())
    # If remove_seasonality=True, run the seasonality removal process. If
    # False, return the min-max normalized time series
    if not remove_seasonality:
        return series_normalized
    else:
        # Take the median of every day of the year across all years in the
        # data, and use this as the seasonality of the time series
        month_values = series.index.month
        day_values = series.index.day
        series_seasonality = series_normalized.groupby([month_values,
                                                        day_values])\
            .transform("median")
        # Remove seasonality from the time series
        return (series_normalized - series_seasonality)


def detect_data_shifts(series,
                       filtering=True, use_default_models=True,
                       method=None, cost=None, penalty=40):
    """
    Detect data shifts in a time series of daily values.

    .. warning:: If the passed time series is less than 2 years in length,
        it will not be corrected for seasonality. Data shift detection will
        be run on the min-max normalized time series with no seasonality
        correction.

    Parameters
    ----------
    series : Pandas series with datetime index.
        Time series of daily PV data values, which can include irradiance
        and power data.
    filtering : Boolean, default True.
        Whether or not to filter out outliers and stale data from the time
        series. If True, then this data is filtered out before running the
        data shift detection sequence. If False, this data is not filtered
        out. Default set to True.
    use_default_models: Boolean, default True
        If True, then default change point detection search parameters are
        used. For time series shorter than 2 years in length, the search
        function is `rpt.Window`  with `model='rbf'`, `width=50` and
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
        Series of boolean values with the input Series' datetime index, where
        detected changepoints are labeled as True, and all other values are
        labeled as False.

    References
    -------
    .. [1] Perry K., and Muller, M. "Automated shift detection in sensor-based
       PV power and irradiance time series", 2022 IEEE 48th Photovoltaic
       Specialists Conference (PVSC).
    """
    try:
        import ruptures as rpt
    except ImportError:
        raise ImportError("data_shifts() requires ruptures.")
    # Run data checks on cleaned data to make sure that the data can be run
    # successfully through the routine
    _run_data_checks(series)
    # Run the filtering sequence, if marked as True
    if filtering:
        series_filtered = _erroneous_filter(series)
    # Drop any duplicated data from the time series
    series_filtered = series_filtered.drop_duplicates()
    # Check if the time series is more than 2 years long. If so, remove
    # seasonality. If not, run analysis on the normalized time series
    if (series_filtered.index.max() -
            series_filtered.index.min()).days <= 730:
        series_processed = _preprocess_data(series_filtered,
                                            remove_seasonality=False)
        seasonality_rmv = False
    else:
        # Perform pre-processing on the time series, to get the
        # seasonality-removed time series.
        series_processed = _preprocess_data(series_filtered,
                                            remove_seasonality=True)
        seasonality_rmv = True
    points = np.array(series_processed.dropna())
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
    # Return a list of dates where changepoints are detected
    series_processed.index.name = "datetime"
    mask = pd.Series(False, index=series_processed.index)
    mask.iloc[result] = True
    # Re-index the mask to include any timestamps that were
    # filtered out as outliers
    mask = mask.reindex(series.index, fill_value=False)
    return mask


def get_longest_shift_segment_dates(series,
                                    filtering=True,
                                    use_default_models=True,
                                    method=None, cost=None,
                                    penalty=40, buffer_day_length=7):
    """
    Return the start and end dates of the longest serially complete time
    series segment.

    During this process, data shift detection is performed, and the
    longest time series segment between changepoints is identified, and the
    start and end dates of that segment are returned, with a settable buffer
    period added to the start date and subtracted from the end date,
    to allow for the segment to stabilize (this helps if the
    changepoint is detected a few days early or a few days late,
    compared to the actual shift date).

    Parameters
    ----------
    series : Pandas series with datetime index.
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
        function is `rpt.Window`  with `model='rbf'`, `width=50` and
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
    buffer_day_length: int, default 7
        Number of days to add to the start date and subtract from the
        end date of the longest detected data shift-free period. This
        buffer period helps to filter out any data that doesn't fit
        within the current data segment. This issue occurs when the
        changepoint is detected a few days early or late compared
        to the actual data shift date.

    Returns
    -------
    start_date: Pandas datetime
        Start date of the longest continuous time series segment that is
        free of data shifts.
    end_date: Pandas datetime
        End date of the longest continuous time series segment that is
        free of data shifts.

    References
    -------
    .. [1] Perry K., and Muller, M. "Automated shift detection in sensor-based
       PV power and irradiance time series", 2022 IEEE 48th Photovoltaic
       Specialists Conference (PVSC). Submitted.
    """
    # Detect indices where data shifts occur
    cpd_mask = detect_data_shifts(series, filtering,
                                  use_default_models,
                                  method, cost, penalty)
    interval_id = cpd_mask.cumsum()
    longest_interval_id = interval_id.value_counts().idxmax()
    index = interval_id.index[interval_id == longest_interval_id]
    # Add a week-long buffer for the start and end dates
    start_date = index.min() + pd.DateOffset(days=buffer_day_length)
    end_date = index.max() - pd.DateOffset(days=buffer_day_length)
    return start_date, end_date
