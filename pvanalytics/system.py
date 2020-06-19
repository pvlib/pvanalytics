"""Functions for identifying system characteristics."""
import pandas as pd
import pvlib
from pvanalytics.util import _fit, _group


def _peak_times(data):
    return pd.DatetimeIndex(
        _group.by_day(data).apply(_fit.quadratic_idxmax),
        tz=data.index.tz
    )


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
    modeled_azimuth = solar_azimuth.reindex(
        peak_times,
        method='bfill'
    )
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
            poa_azimuths = solar_azimuth.reindex(
                _peak_times(poa[solar_zenith < 70]),
                method='bfill'
            )
            sum_of_squares = sum(
                (poa_azimuths.values - modeled_azimuth.values)**2
            )
            if (smallest_sse is None) or (smallest_sse > sum_of_squares):
                smallest_sse = sum_of_squares
                best_azimuth = azimuth
                best_tilt = tilt
    return best_azimuth, best_tilt
