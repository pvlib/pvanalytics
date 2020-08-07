"""Functions for identifying system characteristics."""
import pandas as pd
import numpy as np
import pvlib
from pvanalytics.util import _fit, _group


def _peak_times(data):
    minute_of_day = pd.Series(
        data.index.hour * 60 + data.index.minute,
        index=data.index
    )
    peak_minutes = _group.by_day(data).apply(
        lambda day: pd.Timedelta(
            minutes=_fit.quadratic_idxmax(
                x=minute_of_day[day.index],
                y=day,
                model_range=range(0, 1440)
            )
        )
    )
    return pd.DatetimeIndex(
        np.unique(data.index.date),
        tz=data.index.tz
    ) + peak_minutes


def orientation(power_or_poa, daytime, tilts, azimuths,
                solar_azimuth, solar_zenith, ghi, dhi, dni):
    """Determine system azimuth and tilt from power or POA.

    Parameters
    ----------
    power_or_poa : Series
        Timezone localized series of power or POA irradiance
        measurements.
    sunny : Series
        Boolean series with True for values when it is sunny.
    tilts : list of floats
        list of tilts to check
    azimuths : list of floats
        list of azimuths
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

    """
    peak_times = _peak_times(power_or_poa[daytime])
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
                _peak_times(poa[solar_zenith < 70])
            ]
            sum_of_squares = sum(
                (poa_azimuths[modeled_azimuth.index].values
                 - modeled_azimuth.values)**2
            )
            if (smallest_sse is None) or (smallest_sse > sum_of_squares):
                smallest_sse = sum_of_squares
                best_azimuth = azimuth
                best_tilt = tilt
    return best_azimuth, best_tilt
