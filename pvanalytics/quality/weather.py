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


def relative_humidity_limits(relative_humidity, limits=(0, 100)):
    """Check for extremes in relative humidity data.

    Parameters
    ----------
    relative_humidity : Series
        Relative humidity in %.
    limits : tuple, default (0, 100)
        (lower bound, upper bound) for relative humidity.

    Returns
    -------
    Series
        True if `relative_humidity` >= lower bound and
        `relative_humidity` <= upper_bound.

    """
    return util.check_limits(
        relative_humidity,
        lower_bound=limits[0],
        upper_bound=limits[1],
        inclusive_lower=True,
        inclusive_upper=True
    )
