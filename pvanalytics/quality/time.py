"""Quality tests related to time."""
import pandas as pd


def spacing(times, freq):
    """Check that the spacing between `times` conforms to `freq`.

    Parameters
    ----------
    times : DatetimeIndex
    freq : string or Timedelta
        Expected frequency of `times`.

    Returns
    -------
    Series
        True when the difference between one time and the time before
        it conforms to `freq`.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    if not isinstance(freq, pd.Timedelta):
        freq = pd.Timedelta(freq)
    delta = times.to_series().diff()
    # The first value will be NaT, replace it with freq so the first
    # timestamp is considered valid.
    delta.iloc[0] = freq
    return delta == freq


def _has_dst(event_times, date, window, min_difference):
    before = event_times[date - window:date - pd.Timedelta(days=1)]
    after = event_times[date:date + window]
    before = before.dt.hour * 60 + before.dt.minute
    after = after.dt.hour * 60 + after.dt.minute
    return abs(before.mean() - after.mean()) > min_difference


def has_dst(event_times, shift_dates, window=7, min_difference=45):
    """Return true if `daynight_mask` appears to have daylight-savings shifts
    at the dates in `shift_dates`.

    Compares the mean event time in minutes since midnight over the
    `window` days before and after each date in `shift_dates`. If the
    difference is greater than `min_difference` then a shift has occurred
    on that date.

    Parameters
    ----------
    event_times : Series
        Series with one timestamp for each day. The timestamp should
        correspond to an event that occurs at roughly the same time on
        each day, and shifts with daylight savings transitions. For example,
        you may pass sunrise, sunset, or solar transit time.
    shift_dates : list of datetime-like
        Dates of expected daylight savings time shifts. Can be any type
        that can be converted to a ``pandas.Timestamp`` by
        :py:func:`pandas.to_datetime`.
    window : int, default 7
        Number of days before and after the shift date to consider. When
        passing rounded timestamps in `event_times` it may be necessary to
        use a smaller window. [days]
    min_difference : int, default 45
        Minimum difference between the mean event time before the shift
        date and the mean event time after the event time. If the difference
        is greater than `min_difference` a shift has occurred on that date.
        [minutes]

    Returns
    -------
    list of bool
        Boolean indicating whether a DST shift was found at each date in
        `shift_dates`.
    """
    shift_dates = [pd.to_datetime(date) for date in shift_dates]
    window = pd.Timedelta(days=window)
    return [_has_dst(event_times, date, window, min_difference)
            for date in shift_dates]
