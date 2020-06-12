"""Functions for identifying system orientation."""
import pandas as pd
from pvanalytics.util import _fit


def _conditional_fit(day, fitfunc, freq, default=0.0, min_hours=0.0,
                     peak_min=None):
    # If there are at least `min_hours` of data in `day` and the
    # maximum for the day is greater than `peak_min` then `fitfunc` is
    # applied to fit a curve to the data. `fitfunc` must be a function
    # that takes a Series and returns the :math:`r^2` for a curve fit.
    # If the two conditions are not met then `default` is returned and
    # no curve fitting is performed.
    high_enough = True
    if peak_min is not None:
        high_enough = day.max() > peak_min
    if (_hours(day, freq) > min_hours) and high_enough:
        return fitfunc(day)
    return default


def _freqstr_to_hours(freq):
    # Convert pandas freqstr to hours (as a float)
    if freq.isalpha():
        freq = '1' + freq
    return pd.to_timedelta(freq).seconds / 60


def _hours(data, freq):
    # Return the number of hours in `data` with timestamp
    # spacing given by `freq`.
    return data.count() * _freqstr_to_hours(freq)


def _group_by_day(data):
    # Group data by timezone localized date.
    #
    # We use this function (rather than `data.resample('D')`) because
    # Resampler.apply() makes the data tz-naive when passed to the
    # function being applied. This causes the curve fitting functions
    # to fail when the minute-of-the-day is used as the x-coordinate
    # since removing timezone information can cause data for a day to
    # span two dates.
    #
    # The date needs to be localized in the same timezone as
    # data.index so we can easily reindex the series with the original
    # index at a later time.
    return data.groupby(
        pd.to_datetime(data.index.date).tz_localize(data.index.tz)
    )


def tracking(power_or_irradiance, daytime, correlation_min=0.94,
             fixed_max=0.96, min_hours=5, peak_min=None,
             late_morning='08:45', early_afternoon='17:15'):
    """Flag days where the data matches the profile of a single-axis tracking PV system.

    Tracking days are identified by fitting a restricted quartic to
    the data for each day. If the :math:`r^2` for the fit is greater
    than `correlation_min` and the :math:`r^2` for a quadratic fit is
    less than `fixed_max` then the data on that day is all marked
    True. Data on days where the tracker is stuck, or there is
    substantial variability in the data (i.e. caused by cloudiness) is
    marked False.

    Parameters
    ----------
    power_or_irradiance : Series
        Timezone localized series of power or irradiance measurements.
    daytime : Series
        Boolean series with True for times that are during the
        day. For best results this mask should exclude early morning
        and late afternoon as well as night. These times may have
        problems with shadows that interfere with curve fitting, but
        do not necessarily indicate that the tracker is not working.
    correlation_min : float, default 0.94
        Minimum :math:`r^2` for a day to be considered sunny.
    fixed_max : float, default 0.96
        Maximum :math:`r^2` for a quadratic fit, if the quadratic fit
        is better than this, then tracking/fixed cannot be determined
        and the day is marked False.
    min_hours : float, default 5.0
        Minimum number of hours with data to attempt a fit on a day.
    peak_min : float, default None
        The maximum value for a day must be greater than `peak_min`
        for a fit to be attempted. If the maximum for a day is less
        than `peak_min` then the day will be marked False.
    late_morning : str, default '08:45'
        The earliest time to include in quadratic fits when checking
        for stuck trackers.
    early_afternoon : str, default '17:15'
        The latest time to include in quadtratic fits when checking
        for stuck trackers.

    Returns
    -------
    Series
        Boolean series with True for every value on a day that has a
        tracking profile (see criteria above).

    Notes
    -----
    This algorithm is based on the PVFleets QA Analysis
    project. Copyright (c) 2020 Alliance for Sustainable Energy, LLC.

    """
    freq = pd.infer_freq(power_or_irradiance.index)
    daily_data = _group_by_day(power_or_irradiance[daytime])
    tracking_days = daily_data.apply(
        _conditional_fit,
        _fit.quartic,
        freq=freq,
        min_hours=min_hours,
        peak_min=peak_min
    )
    fixed_days = _group_by_day(
        power_or_irradiance[daytime].between_time(
            late_morning, early_afternoon
        )
    ).apply(
        _conditional_fit,
        _fit.quadratic,
        freq=freq,
        min_hours=min_hours,
        peak_min=peak_min
    )
    return (
        (tracking_days > correlation_min)
        & (tracking_days > fixed_days)
        & (fixed_days < fixed_max)
    ).reindex(power_or_irradiance.index, method='pad', fill_value=False)


def fixed(power_or_irradiance, daytime, correlation_min=0.94,
          min_hours=5, peak_min=None):
    """Flag days where the data matches the profile of a fixed PV system.

    Fixed days are identified when the :math:`r^2` for a quadratic fit
    to the power data is greater than `correlation_min`.

    Parameters
    ----------
    power_or_irradiance : Series
        Timezone localized series of power or irradiance measurements.
    daytime : Series
        Boolean series with True for times that are during the
        day. For best results this mask should exclude early morning
        and evening as well as night. Morning and evening may have
        problems with shadows that interfere with curve fitting, but
        do not necessarily indicate that the day was not sunny.
    correlation_min : float, default 0.94
        Minimum :math:`r^2` for a day to be considered sunny.
    min_hours : float, default 5.0
        Minimum number of hours with data to attempt a fit on a day.
    peak_min : float, default None
        The maximum value for a day must be greater than `peak_min`
        for a fit to be attempted. If the maximum for a day is less
        than `peak_min` then the day will be marked False.

    Returns
    -------
    Series
        True for values on days where `power_or_irradiance` matches
        the expected parabolic profile for a fixed PV system.

    Notes
    -----
    This algorithm is based on the PVFleets QA Analysis
    project. Copyright (c) 2020 Alliance for Sustainable Energy, LLC.

    """
    freq = pd.infer_freq(power_or_irradiance.index)
    daily_data = _group_by_day(
        power_or_irradiance[daytime]
    )
    fixed_days = daily_data.apply(
        _conditional_fit,
        _fit.quadratic,
        freq=freq,
        min_hours=min_hours,
        peak_min=peak_min
    )
    return (
        fixed_days > correlation_min
    ).reindex(power_or_irradiance.index, method='pad', fill_value=False)
