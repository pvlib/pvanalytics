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


def valid_between(series, days=10, minimum_hours=7.75, freq=None):
    """Get the start and end of valid data.

    The start and end dates returned by this function can be used to
    remove large periods of missing data from the begining and end of
    the series. The valid data begins when there are `days`
    consecutive days with data covering at least `minimum_hours` on
    each day. Valid data ends on the last day with `days` consecutive
    days with data covering at least `minimum_hours` preceeding it.

    Parameters
    ----------
    series : Series
        A datetime indexed series.
    days : int
        The minimum number of consecutive valid days for data to be
        considered valid.
    minimum_hours : float
        The number of hours that must have valid data for a day to be
        considered valid.
    freq : string or None, default None
        The frequency to the series. If None, then frequescy is
        inferred from the index.

    Returns
    -------
    start : Datetime or None
        The first valid day. If there are no sufficiently long periods
        of valid days then None is returned.
    end : Datetime or None
        The last valid day. None if start is None.

    Notes
    -----
    This function was derived from the pvfleets_qa_analysis project,
    Copyright (c) 2020 Alliance for Sustainable Energy, LLC. See the
    file LICENSES/PVFLEETS_QA_LICENSE at the top level directory of
    this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/PVFLEETS_QA_LICENSE>`_ for more
    information.

    """
    freq_hours = (pd.Timedelta(freq or pd.infer_freq(series.index)).seconds
                  / (60.0*60.0))
    daily_hours = (series.dropna().resample('D').count()*freq_hours)
    good_days_preceeding = daily_hours[daily_hours >= minimum_hours].rolling(
        str(days)+'D', closed='right'
    ).count()
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
    """Mask out missing data from the begining and end of the data.

    Missing data is determined by the criteria in
    :py:func:`valid_between`.

    Parameters
    ----------
    series : Series
        A DatatimeIndexed series.
    kwargs :
        Any of the keyword arguments that can be passed to
        :py:func:`valid_between`.

    Returns
    -------
    Series
      A series of booleans whith the same index as `series` with False
      up to the first good day, True from the first to the last good
      day, and False from the last good day to the end.

    """
    start, end = valid_between(series, **kwargs)
    s = pd.Series(index=series.index, dtype='bool')
    s.loc[:] = False
    if start:
        s.loc[start.date():end.date()] = True
    return s
