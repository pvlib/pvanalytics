"""Functions for PV system-level metrics."""

from pvlib.temperature import sapm_cell
from pvlib.pvsystem import pvwatts_dc
from dataclasses import dataclass, field
from typing import Union
import numpy as np
import pandas as pd


def _performance_ratio(measured, modeled):
    """ Returns the ratio sum(measured) / sum(modeled)
    """
    return measured.sum() / modeled.sum()


def _calc_cell_temp_weighted(cell_temperature, irradiance):
    """ Returns the ratio sum(cell_temperature * irradiance) / sum(irradiance)
    """
    numerator = cell_temperature * irradiance
    return numerator.sum() / irradiance.sum()


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

    tavg = _calc_cell_temp_weighted(cell_temperature, poa_global)

    modeled = pvwatts_dc(poa_global, cell_temperature, pdc0, gamma_pdc,
                         temp_ref=tavg)

    return _performance_ratio(pac, modeled)


def _calc_pathlength(signal, freq):
    # utility function to calculate the arc length of a time series.
    # used when calculating the variability index.
    dt = signal.index.to_series(keep_tz=True).diff().dt.total_seconds()/60
    dy = signal.diff()
    d = (dy**2 + dt**2)**0.5
    if freq is not None:
        return d.resample(freq).sum()
    return d.sum()


def variability_index(measured, clearsky, freq=None):
    """
    Calculate the variability index.

    Parameters
    ----------
    measured : Series
        Time series of measured GHI. [W/m2]
    clearsky : Series
        Time series of the expected clearsky GHI. [W/m2]
    freq : pandas datetime offset, optional
        Aggregation period (e.g. 'D' for daily).  If not specified,
        the variability index for the entire time series will be returned.

    Returns
    -------
    vi : Series or float
        The calculated variability index

    References
    ----------
    .. [1] Stein, Joshua, Hansen, Clifford, and Reno, Matthew J. THE
       VARIABILITY INDEX: A NEW AND NOVEL METRIC FOR QUANTIFYING IRRADIANCE
       AND PV OUTPUT VARIABILITY. SAND2012-2088C, World Renewable Energy Forum,
       2012.
    """
    return _calc_pathlength(measured, freq) / _calc_pathlength(clearsky, freq)


@dataclass
class BenchmarkingMetrics:

    # Define inputs and their possible types
    measured: Union[pd.Series, np.array] = field(default=None)
    modeled: Union[pd.Series, np.array] = field(default=None)

    def __post_init__(self):
        self.N = len(self.measured)

    # Absolute Mean Bias Deviation (aMBD).
    def MBD(self):
        return (self.modeled - self.measured).sum()

    # Relative Mean Bias Deviation (rMBD)
    def rMBD(self):
        return (self.modeled - self.measured).sum() / self.measured.mean()

    # Absolute Root Mean Square Deviation (aRMSD)
    def RMSD(self):
        return np.sqrt(((self.modeled - self.measured)**2).sum() / self.N)

    # Relative Root Mean Square Deviation (rRMSD)
    def rRMSD(self):
        return (np.sqrt(((self.modeled - self.measured)**2).sum() / self.N)
                / self.measured.mean())

    # Absolute Mean Absolute Deviation (aMAD)
    def aMAD(self):
        return np.abs(self.modeled - self.measured).sum()

    # Absolute Mean Absolute Deviation (aMAD)
    def rMAD(self):
        return np.abs(self.modeled-self.measured).sum() / self.measured.mean()

    # Absolute Standard Deviation (SD)
    # By default Pandas uses ddof=1 and Numpy uses ddof=0
    def aSD(self):
        return np.sqrt(
            (self.N*(self.modeled-self.measured)**2).sum()
            -((self.modeled-self.measured).sum())**2)/self.N

    ## Relative Standard Deviation (SD)
    def rSD(self):
        return self.aSD() / self.measured.mean()

    # Coefficient of determination (R^2)
    def R_squared(self):
        return (
            ((self.modeled-self.modeled.mean()) *
             (self.measured-self.measured.mean())).sum() /
            ((self.modeled-self.modeled.mean())**2 *
             (self.measured-self.measured.mean())**2))**2
