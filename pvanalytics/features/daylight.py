from pvanalytics.util import _fit


def _is_sunny(power_day, correlation_min, tracking):
    # power_day is the power for a single day
    if tracking:
        rsquared = _fit.quartic(power_day)
    else:
        rsquared, _ = _fit.quadratic(power_day)
    return rsquared > correlation_min


def sunny_days(power_or_irradiance, daytime, correlation_min, tracking=False):
    """Return True for values on days that are sunny.

    Sunny days are identified when the :math:`r^2` for a quadratic fit
    to the power data is greater than `correlation_min`. If `tracking`
    is True then a quartic is fit instead of a quadratic.

    """
    power_or_irradiance.resample('D').apply(
        lambda day: _is_sunny(day, correlation_min, tracking)
    )
