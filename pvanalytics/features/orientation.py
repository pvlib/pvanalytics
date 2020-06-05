"""Functions for identifying system orientation."""
import pandas as pd
from pvanalytics.util import _fit


def _filter_low(data, multiplier):
    positive_data = data[data > 0]
    return positive_data[positive_data > multiplier * data.mean()]


def tracking(power_or_irradiance, daytime, correlation_min=0.94,
             fixed_max=0.96, power_min=0.1, late_morning='08:45',
             early_afternoon='17:15'):
    """Flag days where the data matches the profile of a tracking PV system.

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
        and evening as well as night. Morning and evening may have
        problems with shadows that interfere with curve fitting, but
        do not necessarily indicate that the day was not sunny.
    correlation_min : float, default 0.94
        Minimum :math:`r^2` for a day to be considered sunny.
    fixed_max : float, default 0.96
        Maximum :math:`r^2` for a quadratic fit, if the quadratic fit
        is better than this, then tracking/fixed cannot be determined
        and the day is marked False.
    power_min : float, default 0.1
        Data less than `power_min * power_or_irradiance.mean()` is
        removed.
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

    """
    positive_mean = power_or_irradiance[power_or_irradiance > 0].mean()
    high_data = _filter_low(power_or_irradiance[daytime], power_min)
    daily_data = high_data.resample('D')
    tracking = daily_data.apply(
        lambda day: 0.0 if day.max() < positive_mean else _fit.quartic(day)
    )
    fixed = high_data.between_time(
        late_morning, early_afternoon
    ).resample('D').apply(
        lambda day: 0.0 if day.max() < positive_mean else _fit.quadratic(day)
    )
    return (
        (tracking > correlation_min)
        & (tracking > fixed)
        & (fixed < fixed_max)
    ).reindex(power_or_irradiance.index, method='pad', fill_value=False)


def fixed(power_or_irradiance, daytime, correlation_min=0.94,
          power_min=0.4):
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
    power_min : float, default 0.1
        Data less than `power_min * power_or_irradiance.mean()` is
        removed.

    Returns
    -------
    Series
        True for values on days where `power_or_irradiance` matches
        the expected parabolic profile for a fixed PV system.

    """
    daily_data = _filter_low(
        power_or_irradiance[daytime], power_min
    ).resample('D')
    fixed = daily_data.apply(_fit.quadratic)
    return (
        fixed > correlation_min
    ).reindex(power_or_irradiance.index, method='pad', fill_value=False)
