import numpy as np


def get_horizon_mask(horizon, azimuth, elevation):

    """
    Determines if a given (azimuth, elevation) pair is above a horizon profile.

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
    out = elevation >= yp
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
    conditions. For example, the measured irradiance could be obtained with a
    heated plane-of-array pyranometer. When necessary, the irradiance should be
    adjusted for reflections and spectral content.

    Parameters
    ----------
    measured_e_e : array
        Plane-of-array irradiance absent the effect of snow. [W/m2]
    modeled_e_e : array
        Effective irradiance modeled from measured current at maximum power.
        [W/m2]
    i_mp : array
        Maximum power DC current at the resolution of a single module. [A]

    Returns
    -------
    T : array
        Effective transmission. [unitless] Returns NaN where measured DC
        current is NaN and where measured irradiance is zero. Returns zero
        where measured current is zero. Returns 1 where the ratio between
        measured and modeled irradiance exceeds 1.

    References
    ----------
    .. [1] E. C. Cooper, J. L. Braid and L. Burnham, "Identifying the
       Electrical Signature of Snow in Photovoltaic Inverter Data," 2023 IEEE
       50th Photovoltaic Specialists Conference (PVSC), San Juan, PR, USA,
       2023, pp. 1-5. :doi:`10.1109/PVSC48320.2023.10360065`
    """
    T = modeled_e_e / measured_e_e
    # no transmission if no current
    T[np.isnan(i_mp)] = np.nan
    T[i_mp == 0] = 0
    # no transmission if no irradiance
    T[measured_e_e == 0] = np.nan
    # bound T between 0 and 1
    T[T < 0] = np.nan
    T[T > 1] = 1

    return T


def categorize(vmp_ratio, transmission, measured_voltage, modeled_voltage,
               min_dcv, threshold_vratio, threshold_transmission):

    """
    Categorizes electrical behavior into a snow-related mode.

    Modes are defined in [1]_:

    * Mode 0: Indicates periods with enough opaque snow that the system is not
      producing power. Specifically, Mode 0 is when the measured voltage is
      below the inverter's turn-on voltage but the voltage modeled using
      measured irradiance is above the inverter's turn-on voltage.
    * Mode 1: Indicates periods when the system has non-uniform snow and
      both operating voltage and current are decreased. Operating voltage is
      reduced when bypass diodes activate and current is decreased due to
      decreased irradiance.
    * Mode 2: Indicates periods when the operating voltage is reduced but
      current is consistent with snow-free conditions.
    * Mode 3: Indicates periods when the operating voltage is consistent with
      snow-free conditionss but current is reduced.
    * Mode 4: Voltage and current are consistent with snow-free conditions.
    * Mode is None when both measured and voltage modeled from measured
      irradiance are below the inverter turn-on voltage.

    An additional mode was added to cover a case that was not addressed in
    [1]_:
    * Mode -1: Indicates periods where voltage modeled with measured irradiance
      is below the inverter's turn-on voltage. Under Mode -1, it is unknown if
      and how snow impacts power output.

    Parameters
    ----------
    vmp_ratio : array-like
        Ratio between measured voltage and voltage modeled using
        calculated values of transmission. [dimensionless]
    transmission : array-like
        Fraction of plane-of-array irradiance that can reach the array's cells
        through the snow cover. [dimensionless]
    measured_voltage : array-like
        Measured DC voltage. [V]
    modeled_voltage : array-like
        DC voltage modeled using measured plane-of-array irradiance and
        back-of-module temperature. [V]
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
    mode : int or None

    .. [1] E. C. Cooper, J. L. Braid and L. M. Burnham, "Identifying the
       Electrical Signature of Snow in Photovoltaic Inverter Data," 2023 IEEE
       50th Photovoltaic Specialists Conference (PVSC), San Juan, PR, USA,
       2023, pp. 1-5, :doi:`10.1109/PVSC48320.2023.10360065`.
    """
    umin_meas = measured_voltage >= min_dcv
    umin_model = modeled_voltage >= min_dcv

    # offline if model says that voltage is too low.
    # if modeled is above the minimum, then system is
    # possibly generating
    # offline = ~umin_meas & ~umin_model

    # vmp_ratio discrimates between states (1,2) and (3,4)
    uvr = np.where(vmp_ratio >= threshold_vratio, 3, 1)

    # transmission discrimates within (1,2) and (3,4)
    utrans = np.where(transmission >= threshold_transmission, 1, 0)

    # None if nan
    # if offline:
    #   - -1 if umin_model is 0
    # if not offline:
    #   - 0 if umin_meas is 0, i.e., measurement indicate no power but
    #     it must be that umin_model is 1
    #   - state 1, 2, 3, 4 defined by uvr + utrans
    mode = np.where(np.isnan(vmp_ratio) | np.isnan(transmission),
                    None, umin_meas * (uvr + utrans))
    mode = np.where(~umin_model, -1, mode)

    return mode
