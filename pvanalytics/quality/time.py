"""Quality tests related to time."""
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
        See :py:method:`ruptures.detection.Pelt.predict` for more information.

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
