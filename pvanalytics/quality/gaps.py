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
    if freq.isalpha():
        freq = '1' + freq
    delta = pd.to_timedelta(freq)
    return delta.days * (1440 * 60) + delta.seconds


def completeness_score(series, freq=None, keep_index=True):
    """Calculate a data completeness score for each day.

    The completeness score for a given day is the fraction of time in
    the day for which there is data (a value other than NaN). The time
    duration attributed to each value is equal to the timestamp
    spacing of `series`, or `freq` if it is specified. For example, a
    24-hour time series with 30 minute timestamp spacing and 24
    non-NaN values would have data for a total of 12 hours and
    therefore a completeness score of 0.5.

    Parameters
    ----------
    series : Series
        A DatetimeIndexed series.
    freq : str, default None
        Interval between samples in the series as a pandas frequency
        string. If None, the frequency is inferred using
        :py:func:`pandas.infer_freq`.
    keep_index : boolean, default True
        Whether or not the returned series has the same index as
        `series`. If False the returned series will be indexed by day.

    Returns
    -------
    Series
        A series of floats giving the completeness score for each day
        (fraction of the day for which `series` has data).

    Raises
    ------
    ValueError
        If `freq` is longer than the frequency inferred from `series`.

    """
    inferred_seconds = _freq_to_seconds(pd.infer_freq(series.index))
    if freq:
        freq_seconds = _freq_to_seconds(freq)
        seconds_per_sample = freq_seconds
    else:
        seconds_per_sample = inferred_seconds

    if freq and inferred_seconds < freq_seconds:
        raise ValueError("freq must be less than or equal to the"
                         + " frequency of the series")
    daily_counts = series.resample('D').count()
    daily_completeness = (daily_counts * seconds_per_sample) / (1440*60)
    if keep_index:
        return daily_completeness.reindex(series.index, method='pad')
    return daily_completeness


def complete(series, minimum_completeness=0.333, freq=None):
    """Select data points that are part of days with complete data.

    A day has complete data if its completeness score is greater than
    or equal to `minimum_completeness`. The completeness score is
    calculated by :py:func:`completeness_score`.

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
        See :py:func:`completeness_score`.

    See Also
    --------
    :py:func:`completeness_score`

    """
    return completeness_score(series, freq=freq) >= minimum_completeness


def start_stop_dates(series, days=10):
    """Get the start and end of data excluding leading and trailing gaps.

    Parameters
    ----------
    series : Series
        A DatetimeIndexed series of booleans.
    days : int, default 10
        The minimum number of consecutive days where every value in
        `series` is True for data to start or stop.

    Returns
    -------
    start : Datetime or None
        The first valid day. If there are no sufficiently long periods
        of valid days then None is returned.
    stop : Datetime or None
        The last valid day. None if start is None.

    """
    good_days = series.resample('D').apply(all)
    good_days_preceeding = good_days.astype('int').rolling(
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


def trim(series, days=10):
    """Mask the beginning and end of the data if not all True.

    Parameters
    ----------
    series : Series
        A DatetimeIndexed series of booleans
    days : int, default 10
        Minimum number of consecutive days that are all True for
        'good' data to start.

    Returns
    -------
    Series
        A series of booleans with True for all data points between the
        first and last block of `days` consecutive days that are all
        True in `series`. If `series` does not contain such a block of
        consecutive True values, then the returned series will be
        entirely False.

    See Also
    --------
    :py:func:`start_stop_dates`

    """
    start, end = start_stop_dates(series, days=days)
    mask = pd.Series(False, index=series.index)
    if start:
        mask.loc[start.date():end.date()] = True
    return mask


def trim_incomplete(series, minimum_completeness=0.333333, days=10, freq=None):
    """Trim the series based on the completeness score.

    Combines :py:func:`completeness_score` and :py:func:`trim`.

    Parameters
    ----------
    series : Series
        A DatetimeIndexed series.
    minimum_completeness : float, default 0.333333
        The minimum completeness score for each day.
    days : int, default 10
        The number of consecutive days with completeness greater than
        `minumum_completeness` for the 'good' data to start or
        end. See :py:func:`start_stop_dates` for more information.
    freq : str, default None
        The expected frequency of the series. See
        :py:func:`completeness_score` fore more information.

    Returns
    -------
    Series
        A series of booleans with the same index as `series` with
        False up to the first complete day, True between the first and
        the last complete days, and False following the last complete
        day.

    See Also
    --------
    :py:func:`trim`

    :py:func:`completeness_score`

    """
    completeness = completeness_score(series, freq=freq)
    complete_days = completeness >= minimum_completeness
    return trim(complete_days, days=days)
