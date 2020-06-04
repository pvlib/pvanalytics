"""Functions for identifying when the sun is shining."""
from pvanalytics.util import _fit


def sunny_days(power_or_irradiance, daytime,
               correlation_min=0.94, tracking=False):
    """Return True for values on days that are sunny.

    Sunny days are identified when the :math:`r^2` for a quadratic fit
    to the power data is greater than `correlation_min`. If `tracking`
    is True then a quartic is fit instead of a quadratic.

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
    tracking : bool, default False
        Whether the system has a tracker.

    Notes
    -----
    Based on the PVFleets QA Analysis project. Copyright (c) 2020
    Alliance for Sustainable Energy, LLC.

    """
    fit = _fit.quadratic
    if tracking:
        fit = _fit.quartic
    daytime_data = power_or_irradiance[daytime]
    sunny = daytime_data.resample('D').apply(fit)
    # TODO Make sure that when we reindex, Trues don't propagate past
    # midnight if there is a missing day following a sunny day.
    return (sunny > correlation_min).reindex(power_or_irradiance.index, method='pad')
