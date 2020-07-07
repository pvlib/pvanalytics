"""Functions for identifying system orientation."""
import pandas as pd
from pvanalytics.util import _fit, _group


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


def tracking_nrel(power_or_irradiance, daytime, r2_min=0.915,
                  r2_fixed_max=0.96, min_hours=5, peak_min=None,
                  quadratic_mask=None):
    """Flag days that match the profile of a single-axis tracking PV system.

    For the values on a day to be marked True, they must satisfy the
    following three conditions:

    1. a restricted quartic [#]_ must fit the data with :math:`r^2`
       greater than `r2_min`
    2. the :math:`r^2` for the restricted quartic fit must be greater
       than the :math:`r^2` for a quadratic fit
    3. the :math:`r^2` for a quadratic fit must be less than
       `r2_fixed_max`

    Values on days where any one of these conditions is not met are
    marked False.

    .. [#] The specific quartic used for this fit is centered within
       70 minutes of 12:00, the y-value at the center must be within
       15% of the median for the day, and it must open downwards.

    Parameters
    ----------
    power_or_irradiance : Series
        Timezone localized series of power or irradiance measurements.
    daytime : Series
        Boolean series with True for times that are during the
        day. For best results this mask should exclude early morning
        and late afternoon as well as night. Data at these times may have
        problems with shadows that interfere with curve fitting.
    r2_min : float, default 0.915
        Minimum :math:`r^2` for a day to be considered sunny.
    r2_fixed_max : float, default 0.96
        If the :math:`r^2` of the quadratic fit exceeds
        `r2_fixed_max`, then tracking/fixed cannot be distinguished
        and the day is marked False.
    min_hours : float, default 5.0
        Minimum number of hours with data to attempt a fit on a day.
    peak_min : float, default None
        The maximum `power_or_irradiance` value for a day must be
        greater than `peak_min` for a fit to be attempted. If the
        maximum for a day is less than `peak_min` then the day will be
        marked False.
    quadratic_mask : Series, default None
        If None then `daytime` is used. This Series is used to remove
        morning and afternoon times from the data before applying a
        quadratic fit to check for a stuck tracker. It should
        typically exclude more data than `daytime` in order to
        eliminate long tails in the morning or afternoon that appear
        when a tracker is stuck in a West or East orientation.

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
    if quadratic_mask is None:
        quadratic_mask = daytime
    freq = pd.infer_freq(power_or_irradiance.index)
    daily_data = _group.by_day(power_or_irradiance[daytime])
    tracking_days = daily_data.apply(
        _conditional_fit,
        _fit.quartic_restricted,
        freq=freq,
        min_hours=min_hours,
        peak_min=peak_min
    )
    fixed_days = _group.by_day(power_or_irradiance[quadratic_mask]).apply(
        _conditional_fit,
        _fit.quadratic,
        freq=freq,
        min_hours=min_hours,
        peak_min=peak_min
    )
    return (
        (tracking_days > r2_min)
        & (tracking_days > fixed_days)
        & (fixed_days < r2_fixed_max)
    ).reindex(power_or_irradiance.index, method='pad', fill_value=False)


def fixed_nrel(power_or_irradiance, daytime, r2_min=0.94,
               min_hours=5, peak_min=None):
    """Flag days that match the profile of a fixed PV system.

    Fixed days are identified when the :math:`r^2` for a quadratic fit
    to the power data is greater than `r2_min`.

    Parameters
    ----------
    power_or_irradiance : Series
        Timezone localized series of power or irradiance measurements.
    daytime : Series
        Boolean series with True for times that are during the
        day. For best results this mask should exclude early morning
        and evening as well as night. Data at these times may have
        problems with shadows that interfere with curve fitting.
    r2_min : float, default 0.94
        Minimum :math:`r^2` for a day to be considered sunny.
    min_hours : float, default 5.0
        Minimum number of hours with data to attempt a fit on a day.
    peak_min : float, default None
        The maximum `power_or_irradiance` value for a day must be
        greater than `peak_min` for a fit to be attempted. If the
        maximum for a day is less than `peak_min` then the day will be
        marked False.

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
    daily_data = _group.by_day(
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
        fixed_days > r2_min
    ).reindex(power_or_irradiance.index, method='pad', fill_value=False)
