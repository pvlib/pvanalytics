"""Quality tests related to time."""
import warnings
import pandas as pd
import numpy as np
from scipy import stats


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


def _round_multiple(x, to, up_from=None):
    # Return `x` rounded to the nearest multiple of `to`
    #
    # Parameters
    # ----------
    # x : Series
    # to : int
    #     `x` is rounded to a multiple of `to`
    # up_from : int, optional
    #     If the remainder of `x` / `to` is greater than `up_from`
    #     then `x` is rounded up, otherwise `x` is rounded down.
    #     If not specified rounding will go up from `to` // 2.
    up_from = up_from or to // 2
    quotient, remainder = divmod(abs(x), to)
    remainder[remainder > up_from] = to
    remainder[remainder <= up_from] = 0
    return np.sign(x) * (quotient*to + remainder)


def shifts_ruptures(event_times, reference_times, period_min=2,
                    shift_min=15, round_up_from=None,
                    prediction_penalty=13):
    """Identify time shifts using the ruptures library.

    Compares the event time in the expected time zone (`reference_times`)
    with the actual event time in `event_times`.

    The Pelt changepoint detection method is applied to the difference
    between `event_times` and `reference_times`. For each period between
    change points the mode of the difference is rounded to a multiple of
    `shift_min` and returned as the time-shift for all days in that
    period.

    Parameters
    ----------
    event_times : Series
        Time of an event in minutes since midnight. Should be a time series
        of integers with a single value per day. Typically the time mid-way
        between sunrise and sunset.
    reference_times : Series
        Time of event in minutes since midnight for each day in the expected
        timezone. For example, passing solar transit time in a fixed offset
        time zone can be used to detect daylight savings shifts when it is
        unknown whether or not `event_times` is in a fixed offset time zone.
    period_min : int, default 2
        Minimum number of days between shifts. Must be less than or equal to
        the number of days in `event_times`. [days]

        Increasing this parameter will make the result less sensitive to
        transient shifts. For example if your intent is to find and correct
        daylight savings time shifts passing `period_min=60` can give good
        results while excluding shorter periods that appear shifted.
    shift_min : int, default 15
        Minimum shift amount in minutes. All shifts are rounded to a multiple
        of `shift_min`. [minutes]
    round_up_from : int, optional
        The number of minutes greater than a multiple of `shift_min` for a
        shift to be rounded up. If a shift is less than `round_up_from` then
        it will be rounded towards 0. If not specified then the shift will
        be rounded up from `shift_min // 2`. Using a larger value will
        effectively make the shift detection more conservative as small
        variations will tend to be rounded to zero. [minutes]
    prediction_penalty : int, default 13
        Penalty used in assessing change points.
        See :py:meth:`ruptures.detection.Pelt.predict` for more information.

    Returns
    -------
    shifted : Series
        Boolean series indicating whether there appears to be a time
        shift on that day.
    shift_amount : Series
        Time shift in minutes for each day in `event_times`. These times
        can be used to shift the data into the same time zone as
        `reference_times`.

    Raises
    ------
    ValueError
        If the number of days in `event_times` is less than `period_min`.

    Notes
    -----
    Timestamped data from monitored PV systems may not always be localized
    to a consistent timezone. In some cases, data is timestamped with
    local time that may or may not be adjusted for daylight savings time
    transitions. This function helps detect issues of this sort, by
    detecting points where the time of some daily event (e.g. solar noon)
    changes significantly with respect to a reference time for the event.
    If the data's timestamps have not been adjusted for daylight savings
    transitions, the time of day at solar noon will change by roughly 60
    minutes in the days before and after the transition.

    To use this changepoint detection method to determine if your data's
    timestamps involve daylight savings transitions, first reduce your PV
    system data (irradiance or power) to a daily time series, with each
    point being the observed midday time in minutes. For example, if
    sunrise and sunset are inferred from the PV system data, the midday
    time can be inferred as the average of each day's sunrise and sunset
    time of day. To establish the expected midday time, calculate solar
    transit time in time of day.

    Derived from the PVFleets QA project.

    """
    try:
        import ruptures
    except ImportError:
        raise ImportError("time.shifts_ruptures() requires ruptures.")

    if period_min > len(event_times):
        raise ValueError("period_min exceeds number of days in event_times")
    # Drop timezone information. At this point there is one value per day
    # so the timezone is irrelevant.
    time_diff = \
        event_times.tz_localize(None) - reference_times.tz_localize(None)
    break_points = ruptures.Pelt(
        model='rbf',
        jump=1,
        min_size=period_min
    ).fit_predict(
        signal=time_diff.values,
        pen=prediction_penalty
    )
    # Make sure the entire series is covered by the intervals between
    # the breakpoints that were identified above. This means adding a
    # breakpoint at the beginning of the series (0) and at the end if
    # one does not already exist.
    break_points.insert(0, 0)
    if break_points[-1] != len(time_diff):
        break_points.append(len(time_diff))
    time_diff = _round_multiple(time_diff, shift_min, round_up_from)
    shift_amount = time_diff.groupby(
        pd.cut(
            time_diff.reset_index().index,
            break_points,
            include_lowest=True, right=False,
            duplicates='drop'
        )
    ).transform(
        lambda shifted_period: stats.mode(shifted_period).mode[0]
    )
    # localize the shift_amount series to the timezone of the input
    shift_amount = shift_amount.tz_localize(event_times.index.tz)
    return shift_amount != 0, shift_amount


def dst_dates(dates, tz):
    """
    Get the dates on which daylight savings transitions occur in
    the timezone `tz`.

    Parameters
    ----------
    dates : DatetimeIndex
        A datetime index with one timestamp per day.
    tz : str
        A timezone string (e.g. 'America/Denver' or 'CET')

    Returns
    -------
    Series
        Boolean Series with True on days where daylight savings transitions
        occur.
    """
    midnight = pd.Series(
        pd.DatetimeIndex(dates.tz_localize(None).date),
        index=dates
    )
    noon = midnight + pd.Timedelta(hours=12)
    noon = noon.dt.tz_localize(tz)
    dst_shift = noon.apply(lambda t: t.tzinfo.dst(t).total_seconds() / 3600)
    return dst_shift.diff().fillna(0) != 0


def _has_dst(events, date, window, min_difference, missing):
    """Return True if `events` appears to have a daylight savings shift
    on `date`.

    Parameters
    ----------
    events : Series
        Series of timestamps, one per day.
    date : Timestamp
        Timestamp corresponding to midnight on the day to check for daylight
        savings shifts.
    window : Timedelta
        Number of days before and after `date` to use when checking for
        for shifts in `events`
    min_difference : int
        Minimum difference between mean event time before `date` and mean
        event time after `date` for a shift tom be detected.
    missing : str
        Can be 'raise' or 'warn'. Whether to raise an exception or issue
        a warning if there is no data int the window on either side of
        `date`. If 'warn' and there is missing data the False is returned.

    Returns
    -------
    bool
        Whether or not a shift of at least `min_difference` minutes was
        detected in the the event times on `date`.
    """
    before = events[date - window:date - pd.Timedelta(days=1)].dropna()
    after = events[date:date + window - pd.Timedelta(days=1)].dropna()
    if len(before) == 0 or len(after) == 0:
        message = f"No data at {date}. Consider passing a larger `window`."
        if missing == 'raise':
            raise ValueError(message)
        warnings.warn(message)
    before = before.dt.hour * 60 + before.dt.minute
    after = after.dt.hour * 60 + after.dt.minute
    return abs(before.mean() - after.mean()) > min_difference


def has_dst(events, tz, window=7, min_difference=45, missing='raise'):
    """Return True if `events` appears to have daylight savings shifts
    at the dates on which `tz` transitions to or from daylight savings
    time.

    The mean event time in minutes since midnight is calculated
    over the `window` days before and after the date of each daylight
    savings transition in `tz`. For each date, the two mean event times
    (before and after) are compared, and if the difference is greater
    than `min_difference` then a shift has occurred on that date.

    Parameters
    ----------
    events : Series
        Series with one timestamp for each day. The timestamp should
        correspond to an event that occurs at roughly the same time on
        each day. For example,
        you may pass sunrise, sunset, or solar transit time. `events` need
        not be localized.
    tz : str
        Name of a timezone that observes daylight savings and has the same
        or similar UTC offset as the expected time zone for `events`.
    window : int, default 7
        Number of days before and after the shift date to consider. When
        passing rounded timestamps in `events` it may be necessary to
        use a smaller window. [days]
    min_difference : int, default 45
        Minimum difference between the mean event time before the shift
        date and the mean event time after the event time. If the difference
        is greater than `min_difference` a shift has occurred on that date.
        [minutes]
    missing : {'raise', 'warn'}, default 'raise'
        Whether to raise an exception or issue a warning when there is
        no data at a transition date. Can be 'raise' or 'warn'. If 'warn'
        and there is no data adjacent to a transition date, False is
        returned for that date.

    Returns
    -------
    Series
        Boolean Series with the same index as `events` True for dates that
        appear to have daylight savings transitions.

    Raises
    ------
    ValueError
        If there is no data in the `window` days before or after a shift
        date in `events`.
    """
    shift_dates = dst_dates(events.asfreq('D', fill_value=pd.NaT).index, tz)
    # Build a series of transition dates
    shift_dates = shift_dates[shift_dates]
    shift_dates = shift_dates.index.to_series(index=shift_dates.index)
    window = pd.Timedelta(days=window)
    shifted = shift_dates.apply(
        lambda t: _has_dst(
            events,
            pd.Timestamp(t.date(), tz=events.index.tz),
            window,
            min_difference,
            missing
        )
    )
    # pandas 0.23 won't allow .astype('bool') and empty Series with
    # dtype='datetime64[ns]'. Work around this by building a new series
    # of all False.
    if shifted.empty:
        return pd.Series(False, index=events.index)
    return shifted.astype('bool').reindex(events.index, fill_value=False)
