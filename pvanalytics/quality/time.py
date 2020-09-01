"""Quality tests related to time."""
import pandas as pd
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


def _round_fifteen(x):
    # Return `x` rounded to the nearest multiple of 15
    quotient, remainder = divmod(x, 15)
    remainder[remainder > 7] = 15
    remainder[remainder <= 7] = 0
    return quotient*15 + remainder


def shifts_ruptures(daytime, clearsky_midday):
    """Identify time shifts using the ruptures library.

    Parameters
    ----------
    daytime : Series
        Boolean time-series with True for day and False for night.
    clearsky_midday : Series
        Time of midday in minutes for each day with no time shifts
        (i.e.based on solar position for with a fixed-offset time zone).

    Returns
    -------
    Series
        Time shift in minutes at each value in `daytime`.

    Notes
    -----
    Derived from the PVFleets QA project.

    """
    midday = _group.by_day(daytime).apply(
        lambda day: (day[day].index.max() - day[day].index.min()).seconds * 60 / 2
    )
    midday_diff = midday - clearsky_midday
    break_points = ruptures.Pelt(model='rbf').fit_predict(
        signal=midday_diff.values,
        pen=10  # TODO get the correct penalty from Matt
    )
    break_points.insert(0, -1)
    break_points.append(len(midday_diff))
    midday_diff = _round_fifteen(midday_diff)
    shift_amount = midday_diff.groupby(
        pd.cut(
            midday_diff.reset_index().index, break_points, duplicates='drop')
    ).transform(
        lambda shifted_period: stats.mode(shifted_period).mode[0]
    )
    return shift_amount.reindex(daytime.index, method='pad')