import numpy as np


def get_horizon_mask(horizon, azimuth, elevation):

    """
    Determines if a given (azimuth, elevation) pair is above or
    below a horizon profile

    Parameters
    ----------
    horizon : pd.Series
        Series with int index of 0 - 359 (represents azimuth) and float values
        (represents elevation [deg] of horizon profile).
    azimuth : numeric
        Solar azimuth angle. [deg]
    elevation : numeric
        Solar elevation angle. [deg]

    Returns
    -------
    out : bool or NaN
    """
    yp = np.interp(azimuth, horizon.index, horizon.values)
    out = elevation >= yp  # TODO or strictly >
    return out


def get_irradiance_sapm(temp_cell, i_mp, imp0, c0, c1, alpha_imp,
                        irrad_ref=1000, temp_ref=25):
    """
    Model effective irradiance from current at maximum power and cell
    temperature.

    Solves Eqn. 2 from [1]_.

    Parameters
    ----------
    temp_cell : array
        Temperature of cells inside module. [degrees C]
    i_mp : array
        Maximum power current at the resolution of a single module. [A]
    imp0 : float
        Short-circuit current at reference condition. [A]
    c0, c1 : float
        Empirically determined coefficients relating ``i_mp`` to effective
        irradiance.
    alpha_imp : float
        Normalized temperature coefficient for short-circuit current. [1/°C]
    temp_ref : float
        Reference cell temperature. [degrees C]

    Returns
    -------
    effective_irradiance : array
        Effective irradiance. [W/m2]

    References
    ----------
    .. [1] D. L. King, E.E. Boyson, and J.A. Kratochvil, "Photovoltaic Array
       Performance Model", SAND2004-3535, Sandia National Laboratories,
       Albuquerque, NM, 2004.
    """

    a = imp0*c1*(1 + alpha_imp*(temp_cell - temp_ref))
    b = imp0*c0*(1 + alpha_imp*(temp_cell - temp_ref))
    c = -i_mp
    discriminant = np.square(b) - 4*a*c
    effective_irradiance = (-b + np.sqrt(discriminant))/(2*a) * irrad_ref
    return effective_irradiance


def get_irradiance_imp(i_mp, imp0, irrad_ref=1000):
    """
    Model effective irradiance from maximum power current.

    Assumes a linear relationship between effective irradiance and maximum
    power current, i.e., Eqn. 8 from [1]_.

    Parameters
    ----------
    i_mp : array
        Maximum power current at the resolution of a single module. [A]
    imp0 : float
        Short-circuit current at reference condition. [A]

    Returns
    -------
    effective_irradiance : array
        Effective irradiance. [W/m2]

    References
    ----------
    .. [1] C. F. Abe, J. B. Dias, G. Notton, G. A. Faggianelli, G. Pigelet, and
       D. Ouvrard, David, "Estimation of the effective irradiance and bifacial
       gain for PV arrays using the maximum power current", IEEE Journal of
       Photovoltaics, 2022, pp. 432-441. :doi:`10.1109/JPHOTOV.2023.3242117`
    """
    return i_mp / imp0 * irrad_ref


def get_transmission(measured_e_e, modeled_e_e, i_mp):

    """
    Estimate transmittance as the ratio of modeled effective irradiance to
    measured irradiance.

    Measured irradiance should be in the array's plane and represent snow-free
    conditions. When possible, the irradiance should be adjusted for
    reflections and spectral content. For example, the measured irradiance
    could be obtained with a heated plane-of-array pyranometer.

    Parameters
    ----------
    measured_e_e : array
        Plane-of-array irradiance absent the effect of snow. [W/m2]
    modeled_e_e : array
        Effective irradiance modeled from measured current at maximum power.
        [W/m2]
    i_mp : array
        Maximum power current at the resolution of a single module. [A]

    Returns
    -------
    T : array
        Effective transmission. [unitless]

    References
    ----------
    .. [1] E. C. Cooper, J. L. Braid and L. Burnham, "Identifying the
       Electrical Signature of Snow in Photovoltaic Inverter Data," 2023 IEEE
       50th Photovoltaic Specialists Conference (PVSC), San Juan, PR, USA,
       2023, pp. 1-5. :doi:`10.1109/PVSC48320.2023.10360065`
    """

    T = modeled_e_e/measured_e_e
    T[T.isna()] = np.nan
    T[i_mp == 0] = 0
    T[T < 0] = np.nan
    T[T > 1] = 1

    return T


def categorize(vmp_ratio, transmission, voltage, min_dcv,
               threshold_vratio, threshold_transmission):

    """
    Categorizes electrical behavior into a snow-related mode.

    Modes are defined in [1]_:

    * Mode 0: system is covered with enough opaque snow that the system is
      offline due to voltage below the inverter's turn-on voltage.
    * Mode 1: system is online and covered with non-uniform snow, such that
      both operating voltage and current are decreased by the presence of snow.
    * Mode 2: system is online and covered with opaque snow, such that
      operating voltage is decreased by the presence of snow, but transmission
      is consistent with snow-free conditions.
    * Mode 3: system is online and covered with light-transmissive snow, such
      that transmission is decreased but voltage is consistent with all
      system substrings being online.
    * Mode 4: transmisison and voltage are consistent with snow-free
      conditions.

    Parameters
    ----------
    vratio : float
        Ratio between measured voltage and voltage modeled using
        calculated values of transmission. [dimensionless]
    transmission : float
        Fraction of plane-of-array irradiance that can reach the array's cells
        through the snow cover. [dimensionless]
    voltage : float
        [V]
    min_dcv : float
        The lower voltage bound on the inverter's maximum power point
        tracking (MPPT) algorithm. [V]
    threshold_vratio : float
        The lower bound for vratio that is representative of snow-free
        conditions. Determined empirically. Depends on system configuration and
        site conditions. [unitless]
    threshold_transmission : float
        The lower bound on transmission that is found under snow-free
        conditions, determined empirically. [unitless]

    Returns
    -------
    mode : int

    .. [1] E. C. Cooper, J. L. Braid and L. M. Burnham, "Identifying the
       Electrical Signature of Snow in Photovoltaic Inverter Data," 2023 IEEE
       50th Photovoltaic Specialists Conference (PVSC), San Juan, PR, USA,
       2023, pp. 1-5, :doi:`10.1109/PVSC48320.2023.10360065`.
    """

    if np.isnan(vmp_ratio) or np.isnan(transmission):
        return np.nan
    elif voltage < min_dcv:
        return 0
    elif vmp_ratio < threshold_vratio:
        if transmission < threshold_transmission:
            return 1
        elif transmission > threshold_transmission:
            return 2
    elif vmp_ratio > threshold_vratio:
        if transmission < threshold_transmission:
            return 3
        elif transmission > threshold_transmission:
            return 4
    return np.nan
