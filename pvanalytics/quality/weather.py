"""Quality control functions for weather data."""
from pvanalytics.quality import util


def temperature_limits(air_temperature, limits=(-35.0, 50.0)):
    """Identify extreme temperatures.

    Parameters
    ----------
    air_temperature : Series
        Air temperature in Celsius.
    temp_limits : tuple, default (-35, 50)
        (lower bound, upper bound) for temperature.

    Returns
    -------
    Series
        True if `air_temperature` > lower bound and `air_temperature`
        < upper bound.

    """
    return util.check_limits(
        air_temperature, lower_bound=limits[0], upper_bound=limits[1]
    )
