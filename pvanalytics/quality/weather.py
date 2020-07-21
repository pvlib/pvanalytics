"""Quality control functions for weather data."""
from pvanalytics.quality import util
from scipy import stats


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

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

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

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

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

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    return util.check_limits(
        wind_speed,
        lower_bound=limits[0],
        upper_bound=limits[1],
        inclusive_lower=True
    )


def module_temperature_check(module_temperature, irradiance,
                             correlation_min=0.5):
    """Test whether the module temperature is correlated with irradiance.

    Parameters
    ----------
    module_temperature : Series
        Time series of module temperature.
    irradiance : Series
        Time series of irradiance with the same index as
        `module_temperature`. This should be of relatively high
        quality (outliers and other problems removed).
    correlation_min : float, default 0.5
        Minimum correlation between `module_temperature` and
        `irradiance` for the module temperature sensor to 'pass'

    Returns
    -------
    bool
        True if the correlation between `module_temperature` and
        `irradiance` exceeds `correlation_min`.

    """
    _, _, r, _, _ = stats.linregress(module_temperature, irradiance)
    return r > correlation_min
