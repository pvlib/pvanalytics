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
    at or near the dates in `shift_dates`.

    Parameters
    ----------
    event_times : Series
        Series with one timestamp for each day.
    shift_dates : list of datetime-like
        Dates of expected daylight savings time shifts. String should be in
        a format that can be parsed by :py:func:`pandas.to_datetime`.
    window : int
        Number of days before and after the shift date to consider.

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
