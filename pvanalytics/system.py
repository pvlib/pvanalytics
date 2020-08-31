"""Functions for identifying system characteristics."""
import pandas as pd
import numpy as np
import pvlib
from pvlib import solarposition
from pvanalytics.util import _fit, _group


def _peak_times(data):
    minute_of_day = pd.Series(
        data.index.hour * 60 + data.index.minute,
        index=data.index
    )
    peak_minutes = _group.by_day(data).apply(
        lambda day: pd.Timedelta(
            minutes=round(
                _fit.quadratic_vertex(
                    x=minute_of_day[day.index],
                    y=day,
                )
            )
        )
    )
    return pd.DatetimeIndex(
        np.unique(data.index.date),
        tz=data.index.tz
    ) + peak_minutes


def infer_orientation_daily_peak(power_or_poa, sunny, tilts,
                                 azimuths, solar_azimuth,
                                 solar_zenith, ghi, dhi, dni):
    """Determine system azimuth and tilt from power or POA using solar
    azimuth at the daily peak.

    The time of the daily peak is estimated by fitting a quadratic to
    to the data for each day in `power_or_poa` and finding the vertex
    of the fit. A brute force search is performed on clearsky POA
    irradiance for all pairs of candidate azimuths and tilts
    (`azimuths` and `tilts`) to find the pair that results in the
    closest azimuth to the azimuths calculated at the peak times from
    the curve fitting step. Closest is determined by minimizing the
    sum of squared difference between the solar azimuth at the peak
    time in `power_or_poa` and the solar azimuth at maximum clearsky
    POA irradiance.

    The accuracy of the tilt and azimuth returned by this function will
    vary with the time-resolution of the clearsky and solar position
    data. For the best accuracy pass `solar_azimuth`, `solar_zenith`,
    and the clearsky data (`ghi`, `dhi`, and `dni`) with one-minute
    timestamp spacing. If `solar_azimuth` has timestamp spacing less
    than one minute it will be resampled and interpolated to estimate
    azimuth at each minute of the day. Regardless of the timestamp
    spacing these parameters must cover the same days as
    `power_or_poa`.

    Parameters
    ----------
    power_or_poa : Series
        Timezone localized series of power or POA irradiance
        measurements.
    sunny : Series
        Boolean series with True for values during clearsky
        conditions.
    tilts : array-like
        Candidate tilts in degrees.
    azimuths : array-like
        Candidate azimuths in degrees.
    solar_azimuth : Series
        Time series of solar azimuth.
    solar_zenith : Series
        Time series of solar zenith.
    ghi : Series
        Clear sky GHI.
    dhi : Series
        Clear sky DHI.
    dni : Series
        Clear sky DNI.

    Returns
    -------
    azimuth : float
    tilt : float

    Notes
    -----
    Based on PVFleets QA project.

    """
    peak_times = _peak_times(power_or_poa[sunny])
    azimuth_by_minute = solar_azimuth.resample('T').interpolate(
        method='linear'
    )
    modeled_azimuth = azimuth_by_minute[peak_times]
    best_azimuth = None
    best_tilt = None
    smallest_sse = None
    for azimuth in azimuths:
        for tilt in tilts:
            poa = pvlib.irradiance.get_total_irradiance(
                tilt,
                azimuth,
                solar_zenith,
                solar_azimuth,
                ghi=ghi,
                dhi=dhi,
                dni=dni
            ).poa_global
            poa_azimuths = azimuth_by_minute[
                _group.by_day(poa).idxmax()
            ]
            filtered_azimuths = poa_azimuths[np.isin(
                poa_azimuths.index.date,
                modeled_azimuth.index.date
            )]
            sum_of_squares = sum(
                (filtered_azimuths.values - modeled_azimuth.values)**2
            )
            if (smallest_sse is None) or (smallest_sse > sum_of_squares):
                smallest_sse = sum_of_squares
                best_azimuth = azimuth
                best_tilt = tilt
    return best_azimuth, best_tilt


def longitude_solar_noon(solar_noon, utc_offset):
    """Get system longitude from solar noon.

    Uses the method presented in [1]_.

    Parameters
    ----------
    solar_noon : Series
        Time of solar noon (local time in minutes since midnight) on each day.
    utc_offset : int
        Difference between local times zone and UTC (e.g. Denver in is UTC-7).

    Returns
    -------
    float
        Longitude of the system.

    References
    ----------
    .. [1] Haghdadi, N., et al. (2017) A method to estimate the location and
           orientation of distributed photovoltaic systems from their generation
           output data.
    """
    # 720 minutes is noon local time
    time_correction = solar_noon - 720
    eot = solarposition.equation_of_time_pvcdrom(solar_noon.index.dayofyear)
    # calculate the local standard time meridian
    lstm = 15 * utc_offset
    return ((time_correction / 4 - eot/4) + lstm).median()


def infer_orientation_latitude_haghdadi(power, clearsky, longitude,
                                        latitude=None,
                                        tilt_min=0, tilt_max=360,
                                        azimuth_min=0, azimuth_max=360,
                                        latitude_min=-90, latitude_max=90):
    """Infer the tilt, azimuth, and latitude of a PV system.

    Uses the method presented in [1]_.

    Parameters
    ----------
    power : Series
        Time series of power output.
    clearsky : Series
        Boolean series with True for power measurements that occur
        during clear-sky conditions.
    longitude : float
        System longitude.
    latitude : float, optional
        System latitude. If None, latitude will be inferred from `power`.
    tilt_min : float, 0
        Lower bound on acceptable tilt.
    tilt_max : float, 360
        Upper bound on acceptable tilt.
    azimuth_min : float, 0
        Lower bound on acceptable tilt.
    azimuth_max : float, 360
        Upper bound on acceptable tilt.
    latitude_min : float, default -90
        Lower bound on acceptable latitude.
    latitude_max : float, default 90
        Upper bound on acceptable latitude.

    Returns
    -------
    tilt : float
        System tilt in degrees.
    azimuth : float
        System azimuth in degrees.
    latitude : float
        System latitude.

    References
    ----------
    .. [1] Haghdadi, N., et al. (2017) A method to estimate the location and
       orientation of distributed photovoltaic systems from their generation
       output data.
    """
    # TODO
    return 0, 0, 0