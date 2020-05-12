"""Quality control functions for identifying gaps in the data.

Gaps include missing data, interpolation, stuck values, and filler
values (i.e. 999).

"""
import numpy as np
import pandas as pd


def _all_close_to_first(x, rtol=1e-5, atol=1e-8):
    """Test if all values in x are close to x[0].

    Parameters
    ----------
    x : array
    rtol : float, default 1e-5
        Relative tolerance for detecting a change in data values.
    atol : float, default 1e-8
        Absolute tolerance for detecting a change in data values.

    Returns
    -------
    Boolean

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    return np.allclose(a=x, b=x[0], rtol=rtol, atol=atol)


def stale_values_diff(x, window=3, rtol=1e-5, atol=1e-8):
    """Identify stale values in the data.

    For a window of length N, the last value (index N-1) is considered
    stale if all values in the window are close to the first value
    (index 0).

    Parameters
    ----------
    x : Series
        data to be processed
    window : int, default 3
        number of consecutive values which, if unchanged, indicates
        stale data
    rtol : float, default 1e-5
        relative tolerance for detecting a change in data values
    atol : float, default 1e-8
        absolute tolerance for detecting a change in data values

    Parameters rtol and atol have the same meaning as in
    numpy.allclose

    Returns
    -------
    Series
        True for each value that is part of a stale sequence of data

    Raises
    ------
    ValueError
        If window < 2.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    if window < 2:
        raise ValueError('window set to {}, must be at least 2'.format(window))

    flags = x.rolling(window=window).apply(
        _all_close_to_first,
        raw=True,
        kwargs={'rtol': rtol, 'atol': atol}
    ).fillna(False).astype(bool)
    return flags


def interpolation_diff(x, window=3, rtol=1e-5, atol=1e-8):
    """Identify sequences which appear to be linear.

    Sequences are linear if the first difference appears to be
    constant.  For a window of length N, the last value (index N-1) is
    flagged if all values in the window appear to be a line segment.

    Parameters
    ----------
    x : Series
        data to be processed
    window : int, default 3
        number of sequential values that, if the first difference is
        constant, are classified as a linear sequence
    rtol : float, default 1e-5
        tolerance relative to max(abs(x.diff()) for detecting a change
    atol : float, default 1e-8
        absolute tolerance for detecting a change in first difference

    Returns
    -------
    Series
        True for each value that is part of a linear sequence

    Raises
    ------
    ValueError
        If window < 3.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    if window < 3:
        raise ValueError('window set to {}, must be at least 3'.format(window))

    # reduce window by 1 because we're passing the first difference
    flags = stale_values_diff(
        x.diff(periods=1),
        window=window-1,
        rtol=rtol,
        atol=atol
    )
    return flags


def _freq_to_seconds(freq):
    if not freq:
        return None
    if freq.isalpha():
        freq = '1' + freq
    delta = pd.to_timedelta(freq)
    return delta.days * (1440 * 60) + delta.seconds


def daily_completeness(series, freq=None):
    """Calculate a completeness score for each day.

    The completeness score for a given day is the fraction of time in
    the day for which there is data (a value other than NaN). The time
    attributed to each value is equal to the timestamp spacing of
    `series` or `freq` if it is specified. For example, a day with 24
    non-NaN values in a series with 30 minute timestamp spacing would
    have 12 hours of data and therefore completeness score of 0.5.

    Parameters
    ----------
    series : Series
        A DatetimeIndexed series.
    freq : str, default None
        Interval between samples in the series as a pandas frequency
        string. If None, the frequency is inferred using
        :py:func:`pandas.infer_freq`.

    Returns
    -------
    Series
        A series of floats, indexed by day, giving the completeness
        score for each day (fraction of the day for which `series` has
        data).

    Raises
    ------
    ValueError
        If `freq` is longer than the frequency inferred from `series`.

    """
    inferred_seconds = _freq_to_seconds(pd.infer_freq(series.index))
    freq_seconds = _freq_to_seconds(freq)
    seconds_per_sample = freq_seconds or inferred_seconds
    if freq and inferred_seconds < freq_seconds:
        raise ValueError("freq must be less than or equal to the"
                         + " frequency of the series")
    daily_counts = series.resample('D').count()
    return (daily_counts * seconds_per_sample) / (1440*60)


def complete(series, minimum_completeness=0.333, freq=None):
    """Select only data points that are part of a day with complete data.

    A day is complete if its completeness score is greater than or
    equal to `minimum_completeness`. See :py:func:`daily_completeness` for more
    information. For example, a day with 24 non-NaN values in a series
    with 30 minute timestamp spacing would have 12 hours of data and
    therefore a completeness score of 0.5; with the default
    `minimum_completeness=0.333` the day would be marked complete.

    Parameters
    ----------
    series : Series
        The data to be checked for completeness.
    minimum_completeness : float, default 0.333
        Fraction of the day that must have data.
    freq : str, default None
        The expected frequency of the data in `series`. If none then
        the frequency is inferred from the data.

    Returns
    -------
    Series
        A series of booleans with True for each value that is part of
        a day with completeness greater than `minimum_completeness`.

    Raises
    ------
    ValueError
        See :py:func:`daily_completeness`.

    See Also
    --------
    :py:func:`daily_completeness`

    """
    completeness = daily_completeness(series, freq)
    return ((completeness >= minimum_completeness)
            .reindex(series.index, method='pad'))


def start_stop_dates(series, days=10, minimum_completeness=0.333333,
                     freq=None):
    """Get the start and end of data excluding leading and trailing gaps.

    The start and end dates returned by this function can be used to
    remove large periods of missing data from the beginning and end of
    the series. The data starts when there are `days` consecutive days
    with completeness greater than or equal to `minimum_completeness`
    (see :py:func:`daily_completeness`) and ends on the last day with
    `days` consecutive days with completeness at least
    `minimum_completeness` preceeding it. Periods of incomplete days
    between these two dates have no effect on the dates returned.

    Parameters
    ----------
    series : Series
        A DatetimeIndexed series.
    days : int, default 10
        The minimum number of consecutive valid days for data to be
        considered valid.
    minimum_completeness : float, default 0.333333
        The minimum completeness score for a day to be considered
        complete. (see :py:func:`daily_completeness`).
    freq : str or None, default None
        The frequency of data in the series as a pandas frequency
        string. If None, then frequency is inferred from the index.

    Returns
    -------
    start : Datetime or None
        The first valid day. If there are no sufficiently long periods
        of valid days then None is returned.
    stop : Datetime or None
        The last valid day. None if start is None.

    See Also
    --------
    :py:func:`daily_completeness`

    Notes
    -----
    This function was derived from the pvfleets_qa_analysis project,
    Copyright (c) 2020 Alliance for Sustainable Energy, LLC. See the
    file LICENSES/PVFLEETS_QA_LICENSE at the top level directory of
    this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/PVFLEETS_QA_LICENSE>`_ for more
    information.

    """
    completeness = daily_completeness(series, freq)
    complete_days = completeness >= minimum_completeness
    good_days_preceeding = complete_days.astype('int').rolling(
        days, closed='right'
    ).sum()
    good_days_following = good_days_preceeding.shift(periods=-(days-1))
    following_above_threshold = good_days_following[
        good_days_following >= days
    ]
    preceeding_above_threshold = good_days_preceeding[
        good_days_preceeding >= days
    ]

    start = None
    end = None

    if len(following_above_threshold) > 0:
        start = following_above_threshold.index[0]

    if len(preceeding_above_threshold) > 0:
        end = preceeding_above_threshold.index[-1]

    return start, end


def trim(series, **kwargs):
    """Mask out missing data from the beginning and end of the data.

    Removes data preceeding the start date and following the stop date
    returned by :py:func:`start_stop_dates`. If no start and stop
    dates are identified then a series of all False is returned.

    Parameters
    ----------
    series : Series
        A DatetimeIndexed series.
    kwargs :
        Any of the keyword arguments that can be passed to
        :py:func:`start_stop_dates`.

    Returns
    -------
    Series
        A series of booleans with the same index as `series` with
        False up to the first complete day, True between the first and
        the last complete days, and False following the last complete
        day.

    See Also
    --------
    :py:func:`start_stop_dates`

    :py:func:`daily_completeness`

    """
    start, end = start_stop_dates(series, **kwargs)
    s = pd.Series(index=series.index, dtype='bool')
    s.loc[:] = False
    if start:
        s.loc[start.date():end.date()] = True
    return s
