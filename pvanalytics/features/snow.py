import numpy as np


def get_irradiance_sapm(temp_cell, i_mp, imp0, c0, c1, alpha_imp,
                        irrad_ref=1000, temp_ref=25):
    """
    Model effective irradiance from current at maximum power and cell
    temperature.

    Solves Eqn. 2 from [1]_.

    Parameters
    ----------
    temp_cell : array-like
        Temperature of cells inside module. [degrees C]
    i_mp : array-like
        Maximum power current at the resolution of a single module. [A]
    imp0 : float
        Short-circuit current at reference condition. [A]
    c0, c1 : float
        Empirically determined coefficients relating ``i_mp`` to effective
        irradiance.
    alpha_imp : float
        Normalized temperature coefficient for short-circuit current. [1/°C]
    irrad_ref : float, default 1000
        Reference irradiance. [W/m2]
    temp_ref : float, default 25
        Reference cell temperature. [degrees C]

    Returns
    -------
    effective_irradiance : array-like
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
    i_mp : array-like
        Maximum power current at the resolution of a single module. [A]
    imp0 : float
        Short-circuit current at reference condition. [A]
    irrad_ref : float
        The reference irradiance used to determine ``imp0``. [W/m2]

    Returns
    -------
    effective_irradiance : array-like
        Effective irradiance. [W/m2]

    References
    ----------
    .. [1] C. F. Abe, J. B. Dias, G. Notton, G. A. Faggianelli, G. Pigelet,
       and D. Ouvrard, "Estimation of the effective irradiance and bifacial
       gain for PV arrays using the maximum power current", IEEE Journal of
       Photovoltaics, 2022, pp. 432-441. :doi:`10.1109/JPHOTOV.2023.3242117`
    """
    return i_mp / imp0 * irrad_ref


def get_transmission(measured_e_e, modeled_e_e, current_dc):

    """
    Estimate transmission as the ratio of modeled effective irradiance to
    measured irradiance.

    Measured irradiance should be in the array's plane and represent snow-free
    conditions. For example, the measured irradiance could be obtained with a
    heated plane-of-array pyranometer. When necessary, the irradiance should be
    adjusted for reflections and spectral content.

    Parameters
    ----------
    measured_e_e : array-like
        Plane-of-array irradiance without the effect of snow. [W/m2]
    modeled_e_e : array-like
        Effective irradiance modeled from measured current at maximum power.
        [W/m2]
    current_dc : array-like
        Measured DC current at the resolution of a single module. [A]

    Returns
    -------
    T : array-like
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
    transmission = modeled_e_e / measured_e_e
    # no transmission if no current
    transmission[np.isnan(current_dc)] = np.nan
    transmission[current_dc == 0] = 0
    # no transmission if no irradiance
    transmission[measured_e_e == 0] = np.nan
    # bound transmission between 0 and 1
    transmission[transmission < 0] = np.nan
    transmission[transmission > 1] = 1

    return transmission


def categorize(transmission, measured_voltage,
               modeled_voltage_with_snow,
               modeled_voltage_no_snow,
               min_dcv, max_dcv, threshold_vratio, threshold_transmission):

    """
    Categorizes electrical behavior into a snow-related mode.

    Parameters
    ----------
    transmission : array-like
        Fraction of plane-of-array irradiance that can reach the array's cells
        through the snow cover. [dimensionless]
    measured_voltage : array-like
        Measured DC voltage. [V]
    modeled_voltage_with_snow : array-like
        DC voltage modeled using measured plane-of-array irradiance reduced by
        calculated transmission. [V]
    modeled_voltage_no_snow : array-like
        DC voltage modeled using measured plane-of-array irradiance and
        assuming transmission equals 1. [V]
    min_dcv : float
        The lower voltage bound on the inverter's maximum power point
        tracking (MPPT) algorithm. [V]
    max_dcv : float
        The upper voltage bound on the inverter's maximum power point
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
    mode : array-like
        ``mode`` is ``None`` when any of the inputs used to determine ``mode``
        is ``nan``.
    vmp_ratio : array-like
        Ratio between measured DC voltage and DC voltage modeled with
        calculated transmission.

    Notes
    -----
    Modes are defined in [1]_:

    * Mode 0: Indicates periods with enough opaque snow that the system is not
      producing power. Specifically, Mode 0 is when the measured voltage is
      below the lower bound of the inverter's MPPT range but the voltage
      modeled using measured irradiance and ideal transmission is above the
      lower bound of the inverter's MPPT range.
    * Mode 1: Indicates periods when the system has non-uniform snow which
      affects all strings. Mode 1 is assigned when both operating voltage and
      current are reduced. Operating voltage is reduced when snow causes
      mismatch and current is decreased due to reduced transmission.
    * Mode 2: Indicates periods when the system has non-uniform snow which
      causes mismatch for some modules, but doesn't reduce light transmission
      to other modules.
    * Mode 3: Indicates periods when the the system has snow that reduces
      light transmission but doesn't create mismatch. Operating voltage is
      consistent with snow-free conditions but current is reduced.
    * Mode 4: Voltage and current are consistent with snow-free conditions.

    * Mode -1: Indicates periods where it is unknown if or how snow impacts
      power output. Mode -1 includes periods when:

          1. Voltage modeled using measured irradiance and ideal transmission
             is outside the inverter's MPPT range, OR
          2. measured voltage exceeds the upper bound of the inverter's MPPT
             algorithm.

      Mode -1 is added in this function to cover a case that was not addressed
      in [1]_.

    Mode is None when measured values (voltage, current, irradiance or
    temperature) are either missing or produce modeled voltage that is
    invalid.

    References
    ----------
    .. [1] E. C. Cooper, J. L. Braid and L. M. Burnham, "Identifying the
       Electrical Signature of Snow in Photovoltaic Inverter Data," 2023 IEEE
       50th Photovoltaic Specialists Conference (PVSC), San Juan, PR, USA,
       2023, pp. 1-5, :doi:`10.1109/PVSC48320.2023.10360065`.
    """
    umin_meas = measured_voltage >= min_dcv
    umax_meas = measured_voltage < max_dcv

    umin_model = modeled_voltage_no_snow >= min_dcv
    umax_model = modeled_voltage_no_snow < max_dcv

    # Voltage is modeled as NaN if transmission = 0, but V = 0 makes more sense
    modeled_voltage_with_snow_copy = np.where(
        transmission == 0, 0, modeled_voltage_with_snow)

    with np.errstate(divide='ignore'):
        vmp_ratio =\
            measured_voltage /\
            modeled_voltage_with_snow_copy

    # take care of divide by zero
    vmp_ratio = np.where(modeled_voltage_with_snow_copy == 0, 1,
                         vmp_ratio)

    # vmp_ratio discriminates between states (1,2) and (3,4)
    uvr = np.where(vmp_ratio >= threshold_vratio, 3, 1)

    # transmission discriminates within (1,2) and (3,4)
    utrans = np.where(transmission >= threshold_transmission, 1, 0)

    # None if transmission, vmp_ratio, modeled_voltage_no_snow,
    # modeled_voltage_with_snow_copy, or measured_voltage is nan

    # 0 if umin_meas is 0, i.e., measurement indicate no power
    # state 1, 2, 3, 4 defined by uvr + utrans
    mode = umin_meas * (uvr + utrans)

    # -1 if umax_model is 0
    # -1 if umax_meas is 0 or umin_meas is 0
    mode = np.where(~umin_model | ~umax_model | ~umax_meas, -1, mode)

    # replace nan with None
    mode = np.where(np.isnan(vmp_ratio) | np.isnan(transmission) |
                    np.isnan(measured_voltage) |
                    np.isnan(modeled_voltage_with_snow_copy) |
                    np.isnan(modeled_voltage_no_snow),
                    None, mode)

    return mode, vmp_ratio
