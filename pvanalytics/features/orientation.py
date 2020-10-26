"""Functions for identifying system orientation."""
import pandas as pd
from pvanalytics import util
from pvanalytics.util import _fit, _group


def _conditional_fit(day, minutes, fitfunc, freq, default=0.0, min_hours=0.0,
                     peak_min=None):
    # Return the :math:`r^2` of a curve fit to a single day of data if
    # certain conditions are met.
    #
    # `fitfunc` does the curve fitting and is only applied if two
    # conditions are met:
    # - There must be more than `min_hours` of data in `day`
    #   (determined by the number of values in `day` times `freq`).
    # - If `peak_min` is specified then no curve fitting will be
    #   performed unless the maximum value in `day` is at least
    #   `peak_min`.
    #
    # If either condition is not satisfied then `default` is returned.
    #
    # Parameters
    # ----------
    # day : Series
    #     y-values to which `fitfunc` will be applied.
    # minutes : Series
    #     x-values for curve fitting. The index for `x` must be a
    #     superset of the index for `day`.
    # fitfunc : function
    #     Function to perform curve fit. Must accept two parameters,
    #     the x-values and y-values, and return the :math:`r^2`
    #     of the curve fit.
    # freq : str
    #     Timestamp spacing for data in `day`.
    # default : float, default 0.0
    #     Value to be returned if the conditions above are not
    #     satisfied and `fitfunc` is not applied.
    # min_hours : float, default 0.0
    #     Minimum hours in `day` with data for curve fitting to be performed.
    # peak_min : float or None, default None
    #     Maximum value in `day` must be at least `peak_min` for curve
    #     fitting to be performed.
    #
    # Returns
    # -------
    # float
    #     The :math:`r^2` of the curve fit from `fitfunc` or `default`
    #     if fit was not performed.
    high_enough = True
    if peak_min is not None:
        high_enough = day.max() > peak_min
    if (_hours(day, freq) > min_hours) and high_enough:
        return fitfunc(minutes[day.index], day)
    return default


def _freqstr_to_hours(freq):
    # Convert pandas freqstr to hours (as a float)
    return util.freq_to_timedelta(freq).seconds / 3600


def _hours(data, freq):
    # Return the number of hours in `data` with timestamp
    # spacing given by `freq`.
    return data.count() * _freqstr_to_hours(freq)


def tracking_nrel(power_or_irradiance, daytime, r2_min=0.915,
                  r2_fixed_max=0.96, min_hours=5, peak_min=None,
                  quadratic_mask=None):
    """Flag days that match the profile of a single-axis tracking PV system
    on a sunny day.

    This algorithm relies on the observation that the power profile of
    a single-axis tracking PV system tends to resemble a quartic
    polynomial on a sunny day, I.e., two peaks are observed, one
    before and one after the sun crosses the tracker azimuth. By
    contrast, the power profile for a fixed tilt PV system often
    resembles a quadratic polynomial on a sunny day, with a single
    peak when the sun is near the system azimuth.

    The algorithm fits both a quartic and a quadratic polynomial to
    each day's data.  A day is marked True if the quartic fit has a
    sufficiently high :math:`r^2` and the quadratic fit has a
    sufficiently low :math:`r^2`.  Specifically, a day is marked True
    when three conditions are met:

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
        Minimum :math:`r^2` of a quartic fit for a day to be marked True.
    r2_fixed_max : float, default 0.96
        If the :math:`r^2` of a quadratic fit exceeds
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
        quadratic fit. The mask should
        typically exclude more data than `daytime` in order to
        eliminate long tails in the morning or afternoon that can
        appear if a tracker is stuck in a West or East orientation.

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
    minutes = pd.Series(
        power_or_irradiance.index.hour * 60 + power_or_irradiance.index.minute,
        index=power_or_irradiance.index
    )
    daily_data = _group.by_day(power_or_irradiance[daytime])
    tracking_days = daily_data.apply(
        _conditional_fit,
        fitfunc=_fit.quartic_restricted_r2,
        minutes=minutes,
        freq=freq,
        min_hours=min_hours,
        peak_min=peak_min
    )
    fixed_days = _group.by_day(power_or_irradiance[quadratic_mask]).apply(
        _conditional_fit,
        fitfunc=_fit.quadratic_r2,
        minutes=minutes,
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
    """Flag days that match the profile of a fixed PV system on a sunny day.

    This algorithm relies on the observation that the power profile of a
    fixed tilt PV system often resembles a quadratic polynomial on a
    sunny day, with a single peak when the sun is near the system azimuth.

    A day is marked True when the :math:`r^2` for a quadratic fit to the
    power data is greater than `r2_min`.

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
        Minimum :math:`r^2` of a quadratic fit for a day to be marked True.
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
        the expected parabolic profile for a fixed PV system on a sunny day.

    Notes
    -----
    This algorithm is based on the PVFleets QA Analysis
    project. Copyright (c) 2020 Alliance for Sustainable Energy, LLC.

    """
    freq = pd.infer_freq(power_or_irradiance.index)
    daily_data = _group.by_day(
        power_or_irradiance[daytime]
    )
    minutes = pd.Series(
        power_or_irradiance.index.hour * 60 + power_or_irradiance.index.minute,
        index=power_or_irradiance.index
    )
    fixed_days = daily_data.apply(
        _conditional_fit,
        fitfunc=_fit.quadratic_r2,
        minutes=minutes,
        freq=freq,
        min_hours=min_hours,
        peak_min=peak_min
    )
    return (
        fixed_days > r2_min
    ).reindex(power_or_irradiance.index, method='pad', fill_value=False)
