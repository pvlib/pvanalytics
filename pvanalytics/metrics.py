"""Functions for PV system-level metrics."""

from pvlib.temperature import sapm_cell
from pvlib.pvsystem import pvwatts_dc


def performance_ratio_nrel(poa_global, temp_air, wind_speed, pac, pdc0,
                           a=-3.56, b=-0.075, deltaT=3, gamma_pdc=-0.00433):
    r"""
    Calculate NREL Performance Ratio.

    See equation [5] in Weather-Corrected Performance Ratio [1]_ for details
    on the weighted method for Tref.

    Parameters
    ----------
    poa_global : numeric
        Total incident irradiance [W/m^2].

    temp_air : numeric
        Ambient dry bulb temperature [C].

    wind_speed : numeric
        Wind speed at a height of 10 meters [m/s].

    pac : float
        AC power [kW].

    pdc0 : float
        Power of the modules at 1000 W/m2 and cell reference temperature [kW].

    a : float
        Parameter :math:`a` in SAPM model [unitless].

    b : float
        Parameter :math:`b` in in SAPM model [s/m].

    deltaT : float
        Parameter :math:`\Delta T` in SAPM model [C].

    gamma_pdc : float
        The temperature coefficient in units of 1/C. Typically -0.002 to
        -0.005 per degree C [1/C].

    Returns
    -------
    performance_ratio: float
        Performance Ratio of data.

    References
    ----------
    .. [1] Dierauf et al. "Weather-Corrected Performance Ratio". NREL, 2013.
       https://www.nrel.gov/docs/fy13osti/57991.pdf
    """

    cell_temperature = sapm_cell(poa_global, temp_air, wind_speed, a, b,
                                 deltaT)

    tcell_poa_global = poa_global * cell_temperature
    tref = tcell_poa_global.sum() / poa_global.sum()

    pdc = pvwatts_dc(poa_global, cell_temperature, pdc0, gamma_pdc,
                     temp_ref=tref)

    performance_ratio = pac.sum() / pdc.sum()

    return performance_ratio
