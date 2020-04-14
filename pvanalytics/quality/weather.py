"""Quality control functions for weather data."""
from pvanalytics.quality import util


def temperature_limits(air_temperature, limits=(-35.0, 50.0)):
    """Identify temperature values that are within limits.

    Parameters
    ----------
    air_temperature : Series
        Air temperature [C].
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
    """Identify relative humidity values that are within limits.

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


def wind_limits(wind_speed, limits=(0.0, 50.0)):
    """Identify wind speed values that are within limits.

    Parameters
    ----------
    wind_speed : Series
        Wind speed in :math:`m/s`
    wind_limits : tuple, default (0, 50)
        (lower bound, upper bound) for wind speed.

    Returns
    -------
    Series
        True if `wind_speed` >= lower bound and `wind_speed` < upper
        bound.

    """
    return util.check_limits(
        wind_speed,
        lower_bound=limits[0],
        upper_bound=limits[1],
        inclusive_lower=True
    )
