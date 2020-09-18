"""Quality tests related to time."""
import pandas as pd
import numpy as np
import ruptures
from scipy import stats
from pvanalytics.util import _group


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


def shifts_ruptures(daytime, clearsky_midday, period_min=2):
    """Identify time shifts using the ruptures library.

    Parameters
    ----------
    daytime : Series
        Boolean time-series with True for day and False for night.
    clearsky_midday : Series
        Time of midday in minutes for each day with no time shifts
        (i.e.based on solar position for with a fixed-offset time zone).
    period_min : int, default 2
        Minimum number of days between shifts. Must be less than or equal to
        the number of days in `daytime`.

        Increasing this parameter will make the result less sensitive to
        transient shifts. For example if your intent is to find and correct
        daylight savings time shifts passing `period_min=60` can give good
        results while excluding shorter periods that appear shifted.

    Returns
    -------
    Series
        Time shift in minutes at each value in `daytime`.

    Raises
    ------
    ValueError
        If the number of days in `daytime` is less than `period_min`.

    Notes
    -----
    Derived from the PVFleets QA project.

    """
    midday = _group.by_day(daytime).apply(
        lambda day: (day[day].index.min()
                     + ((day[day].index.max() - day[day].index.min()) / 2))
    )
    if period_min > len(midday):
        raise ValueError("period_min exceeds number of days in series")
    midday_minutes = midday.dt.hour * 60 + midday.dt.minute
    midday_diff = midday_minutes - clearsky_midday
    break_points = ruptures.Pelt(
        model='rbf',
        jump=1,
        min_size=period_min
    ).fit_predict(
        signal=midday_diff.values,
        pen=15
    )
    # Make sure the entire series is covered by the intervals between
    # the breakpoints that were identified above. This means adding a
    # breakpoint at the beginning of the series (0) and at the end if
    # one does not already exist.
    break_points.insert(0, 0)
    if break_points[-1] != len(midday_diff):
        break_points.append(len(midday_diff))
    midday_diff = _round_multiple(midday_diff, 15)
    shift_amount = midday_diff.groupby(
        pd.cut(
            midday_diff.reset_index().index,
            break_points,
            include_lowest=True, right=False,
            duplicates='drop'
        )
    ).transform(
        lambda shifted_period: stats.mode(shifted_period).mode[0]
    )
    return shift_amount.reindex(daytime.index, method='pad')
