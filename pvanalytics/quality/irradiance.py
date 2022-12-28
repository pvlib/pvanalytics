"""Quality control functions for irradiance data."""

import numpy as np
import pandas as pd
from scipy import integrate
from pvlib.tools import cosd
import pvlib

from pvanalytics import quality
from pvanalytics import util


QCRAD_LIMITS = {'ghi_ub': {'mult': 1.5, 'exp': 1.2, 'min': 100},
                'dhi_ub': {'mult': 0.95, 'exp': 1.2, 'min': 50},
                'dni_ub': {'mult': 1.0, 'exp': 0.0, 'min': 0},
                'ghi_lb': -4, 'dhi_lb': -4, 'dni_lb': -4}

QCRAD_CONSISTENCY = {
    'ghi_ratio': {
        'low_zenith': {
            'zenith_bounds': [0.0, 75],
            'ghi_bounds': [50, np.Inf],
            'ratio_bounds': [0.92, 1.08]},
        'high_zenith': {
            'zenith_bounds': [75, 93],
            'ghi_bounds': [50, np.Inf],
            'ratio_bounds': [0.85, 1.15]}},
    'dhi_ratio': {
        'low_zenith': {
            'zenith_bounds': [0.0, 75],
            'ghi_bounds': [50, np.Inf],
            'ratio_bounds': [0.0, 1.05]},
        'high_zenith': {
            'zenith_bounds': [75, 93],
            'ghi_bounds': [50, np.Inf],
            'ratio_bounds': [0.0, 1.10]}}}


def _qcrad_ub(dni_extra, sza, lim):
    cosd_sza = cosd(sza)
    cosd_sza[cosd_sza < 0] = 0
    return lim['mult'] * dni_extra * cosd_sza**lim['exp'] + lim['min']


def check_ghi_limits_qcrad(ghi, solar_zenith, dni_extra, limits=None):
    r"""Test for physical limits on GHI using the QCRad criteria.

    Test is applied to each GHI value. A GHI value passes if value >
    lower bound and value < upper bound. Lower bounds are constant for
    all tests. Upper bounds are calculated as

    .. math::
        ub = min + mult * dni\_extra * cos( solar\_zenith)^{exp}

    Parameters
    ----------
    ghi : Series
        Global horizontal irradiance in :math:`W/m^2`
    solar_zenith : Series
        Solar zenith angle in degrees
    dni_extra : Series
        Extraterrestrial normal irradiance in :math:`W/m^2`
    limits : dict, default QCRAD_LIMITS
        Must have keys 'ghi_ub' and 'ghi_lb'. For 'ghi_ub' value is a
        dict with keys {'mult', 'exp', 'min'} and float values. For
        'ghi_lb' value is a float.

    Returns
    -------
    Series
        True where value passes limits test.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    if not limits:
        limits = QCRAD_LIMITS
    ghi_ub = _qcrad_ub(dni_extra, solar_zenith, limits['ghi_ub'])

    ghi_limit_flag = quality.util.check_limits(ghi, limits['ghi_lb'], ghi_ub)

    return ghi_limit_flag


def check_dhi_limits_qcrad(dhi, solar_zenith, dni_extra, limits=None):
    r"""Test for physical limits on DHI using the QCRad criteria.

    Test is applied to each DHI value. A DHI value passes if value >
    lower bound and value < upper bound. Lower bounds are constant for
    all tests. Upper bounds are calculated as

    .. math::
        ub = min + mult * dni\_extra * cos( solar\_zenith)^{exp}

    Parameters
    ----------
    dhi : Series
        Diffuse horizontal irradiance in :math:`W/m^2`
    solar_zenith : Series
        Solar zenith angle in degrees
    dni_extra : Series
        Extraterrestrial normal irradiance in :math:`W/m^2`
    limits : dict, default QCRAD_LIMITS
        Must have keys 'dhi_ub' and 'dhi_lb'. For 'dhi_ub' value is a
        dict with keys {'mult', 'exp', 'min'} and float values. For
        'dhi_lb' value is a float.

    Returns
    -------
    Series
        True where value passes limit test.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    if not limits:
        limits = QCRAD_LIMITS

    dhi_ub = _qcrad_ub(dni_extra, solar_zenith, limits['dhi_ub'])

    dhi_limit_flag = quality.util.check_limits(dhi, limits['dhi_lb'], dhi_ub)

    return dhi_limit_flag


def check_dni_limits_qcrad(dni, solar_zenith, dni_extra, limits=None):
    r"""Test for physical limits on DNI using the QCRad criteria.

    Test is applied to each DNI value. A DNI value passes if value >
    lower bound and value < upper bound. Lower bounds are constant for
    all tests. Upper bounds are calculated as

    .. math::
        ub = min + mult * dni\_extra * cos( solar\_zenith)^{exp}

    Parameters
    ----------
    dni : Series
        Direct normal irradiance in :math:`W/m^2`
    solar_zenith : Series
        Solar zenith angle in degrees
    dni_extra : Series
        Extraterrestrial normal irradiance in :math:`W/m^2`
    limits : dict, default QCRAD_LIMITS
        Must have keys 'dni_ub' and 'dni_lb'. For 'dni_ub' value is a
        dict with keys {'mult', 'exp', 'min'} and float values. For
        'dni_lb' value is a float.

    Returns
    -------
    Series
        True where value passes limit test.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    if not limits:
        limits = QCRAD_LIMITS

    dni_ub = _qcrad_ub(dni_extra, solar_zenith, limits['dni_ub'])

    dni_limit_flag = quality.util.check_limits(dni, limits['dni_lb'], dni_ub)

    return dni_limit_flag


def check_irradiance_limits_qcrad(solar_zenith, dni_extra, ghi=None, dhi=None,
                                  dni=None, limits=None):
    r"""Test for physical limits on GHI, DHI or DNI using the QCRad criteria.

    Criteria from [1]_ are used to determine physically plausible
    lower and upper bounds. Each value is tested and a value passes if
    value > lower bound and value < upper bound. Lower bounds are
    constant for all tests. Upper bounds are calculated as

    .. math::
        ub = min + mult * dni\_extra * cos( solar\_zenith)^{exp}

    .. note:: If any of `ghi`, `dhi`, or `dni` are None, the
       corresponding element of the returned tuple will also be None.

    Parameters
    ----------
    solar_zenith : Series
        Solar zenith angle in degrees
    dni_extra : Series
        Extraterrestrial normal irradiance in :math:`W/m^2`
    ghi : Series or None, default None
        Global horizontal irradiance in :math:`W/m^2`
    dhi : Series or None, default None
        Diffuse horizontal irradiance in :math:`W/m^2`
    dni : Series or None, default None
        Direct normal irradiance in :math:`W/m^2`
    limits : dict, default QCRAD_LIMITS
        for keys 'ghi_ub', 'dhi_ub', 'dni_ub', value is a dict with
        keys {'mult', 'exp', 'min'} and float values. For keys
        'ghi_lb', 'dhi_lb', 'dni_lb', value is a float.

    Returns
    -------
    ghi_limit_flag : Series
        True for each value that is physically possible. None if `ghi` is None.
    dhi_limit_flag : Series
        True for each value that is physically possible. None if `dni` is None.
    dni_limit_flag : Series
        True for each value that is physically possible. None if `dni` is None.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    References
    ----------
    .. [1] C. N. Long and Y. Shi, An Automated Quality Assessment and Control
       Algorithm for Surface Radiation Measurements, The Open Atmospheric
       Science Journal 2, pp. 23-37, 2008.

    """
    if not limits:
        limits = QCRAD_LIMITS

    if ghi is not None:
        ghi_limit_flag = check_ghi_limits_qcrad(ghi, solar_zenith, dni_extra,
                                                limits=limits)
    else:
        ghi_limit_flag = None

    if dhi is not None:
        dhi_limit_flag = check_dhi_limits_qcrad(dhi, solar_zenith, dni_extra,
                                                limits=limits)
    else:
        dhi_limit_flag = None

    if dni is not None:
        dni_limit_flag = check_dni_limits_qcrad(dni, solar_zenith, dni_extra,
                                                limits=limits)
    else:
        dni_limit_flag = None

    return ghi_limit_flag, dhi_limit_flag, dni_limit_flag


def _get_bounds(bounds):
    return (bounds['ghi_bounds'][0], bounds['ghi_bounds'][1],
            bounds['zenith_bounds'][0], bounds['zenith_bounds'][1],
            bounds['ratio_bounds'][0], bounds['ratio_bounds'][1])


def _check_irrad_ratio(ratio, ghi, sza, bounds):
    # unpack bounds dict
    ghi_lb, ghi_ub, sza_lb, sza_ub, ratio_lb, ratio_ub = _get_bounds(bounds)
    # for zenith set inclusive_lower to handle edge cases, e.g., zenith=0
    return (
        quality.util.check_limits(
            sza, lower_bound=sza_lb, upper_bound=sza_ub, inclusive_lower=True)
        & quality.util.check_limits(
            ghi, lower_bound=ghi_lb, upper_bound=ghi_ub)
        & quality.util.check_limits(
            ratio, lower_bound=ratio_lb, upper_bound=ratio_ub)
    )


def check_irradiance_consistency_qcrad(solar_zenith, ghi, dhi, dni,
                                       param=None):
    """Check consistency of GHI, DHI and DNI using QCRad criteria.

    Uses criteria given in [1]_ to validate the ratio of irradiance
    components.

    .. warning:: Not valid for night time. While you can pass data
       from night time to this function, be aware that the truth
       values returned for that data will not be valid.

    Parameters
    ----------
    solar_zenith : Series
        Solar zenith angle in degrees
    ghi : Series
        Global horizontal irradiance in :math:`W/m^2`
    dhi : Series
        Diffuse horizontal irradiance in :math:`W/m^2`
    dni : Series
        Direct normal irradiance in :math:`W/m^2`
    param : dict
        keys are 'ghi_ratio' and 'dhi_ratio'. For each key, value is a dict
        with keys 'high_zenith' and 'low_zenith'; for each of these keys,
        value is a dict with keys 'zenith_bounds', 'ghi_bounds', and
        'ratio_bounds' and value is an ordered pair [lower, upper]
        of float.

    Returns
    -------
    consistent_components : Series
        True where `ghi`, `dhi` and `dni` components are consistent.
    diffuse_ratio_limit : Series
        True where diffuse to GHI ratio passes limit test.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    References
    ----------
    .. [1] C. N. Long and Y. Shi, An Automated Quality Assessment and Control
       Algorithm for Surface Radiation Measurements, The Open Atmospheric
       Science Journal 2, pp. 23-37, 2008.

    """
    if not param:
        param = QCRAD_CONSISTENCY

    # sum of components
    component_sum = dni * cosd(solar_zenith) + dhi
    ghi_ratio = ghi / component_sum
    dhi_ratio = dhi / ghi

    bounds = param['ghi_ratio']
    consistent_components = (
        _check_irrad_ratio(ratio=ghi_ratio, ghi=component_sum,
                           sza=solar_zenith, bounds=bounds['high_zenith'])
        | _check_irrad_ratio(ratio=ghi_ratio, ghi=component_sum,
                             sza=solar_zenith, bounds=bounds['low_zenith']))

    bounds = param['dhi_ratio']
    diffuse_ratio_limit = (
        _check_irrad_ratio(ratio=dhi_ratio, ghi=ghi, sza=solar_zenith,
                           bounds=bounds['high_zenith'])
        | _check_irrad_ratio(ratio=dhi_ratio, ghi=ghi, sza=solar_zenith,
                             bounds=bounds['low_zenith']))

    return consistent_components, diffuse_ratio_limit


def clearsky_limits(measured, clearsky, csi_max=1.1):
    """Identify irradiance values which do not exceed clearsky values.

    Uses :py:func:`pvlib.irradiance.clearsky_index` to compute the
    clearsky index as the ratio of `measured` to `clearsky`. Compares the
    clearsky index to `csi_max` to identify values in `measured` that
    are less than or equal to `csi_max`.

    Parameters
    ----------
    measured : Series
        Measured irradiance in :math:`W/m^2`.
    clearsky : Series
        Expected clearsky irradiance in :math:`W/m^2`.
    csi_max : float, default 1.1
        Maximum ratio of `measured` to `clearsky` (clearsky index).

    Returns
    -------
    Series
        True for each value where the clearsky index is less than or
        equal to `csi_max`.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    csi = pvlib.irradiance.clearsky_index(
        measured,
        clearsky,
        max_clearsky_index=np.Inf
    )
    return quality.util.check_limits(
        csi, upper_bound=csi_max, inclusive_upper=True
    )


def _to_hours(freqstr):
    return util.freq_to_timedelta(freqstr).seconds / 3600


def _daily_total(series):
    # Resample the series returning the integral for each day
    # calculated using the trapezoid rule.
    #
    # If the series has uniform timestamp spacing (i.e. pd.infer_freq
    # returns a value other than None) then the frequency is used as
    # the spacing between each value to speed up integration. If the
    # frequency cannot be inferred, then the hour of the day is
    # calculated for each value and used as the x-coordinate in the
    # integration.
    freq = pd.infer_freq(series.index)
    if freq:
        freq_hours = _to_hours(freq)
        return series.resample('D').apply(
            integrate.trapz,
            dx=freq_hours
        )
    hours = pd.Series(
        (series.index.minute / 60) + series.index.hour,
        index=series.index
    )
    return series.resample('D').apply(
        lambda day: integrate.trapz(y=day, x=hours[day.index])
    )


def daily_insolation_limits(irrad, clearsky, daily_min=0.4, daily_max=1.25):
    """Check that daily insolation lies between minimum and maximum values.

    Irradiance measurements and clear-sky irradiance on each day are
    integrated with the trapezoid rule to calculate daily insolation.

    Parameters
    ----------
    irrad : Series
        Irradiance measurements (GHI or POA).
    clearsky : Series
        Clearsky irradiance.
    daily_min : float, default 0.4
        Minimum ratio of daily insolation to daily clearsky insolation.
    daily_max : float, default 1.25
        Maximum ratio of daily insolation to daily clearsky insolation.

    Returns
    -------
    Series
        True for values on days where the ratio of daily insolation to
        daily clearsky insolation is between `daily_min` and `daily_max`.

    Notes
    -----
    The default limits (`daily_max` and `daily_min`) have been set for
    GHI and POA irradiance for systems with *fixed* azimuth and tilt.
    If you pass POA irradiance for a tracking system it is recommended
    that you increase `daily_max` to 1.35.

    The default values for `daily_min` and `daily_max` were taken from
    the PVFleets QA Analysis project.

    """
    daily_irradiance = _daily_total(irrad)
    daily_clearsky = _daily_total(clearsky)
    good_days = quality.util.check_limits(
        daily_irradiance/daily_clearsky,
        upper_bound=daily_max,
        lower_bound=daily_min
    )
    return good_days.reindex(irrad.index, method='pad', fill_value=False)


def _fill_nighttime(component, component_sum_df, fill_night_value,
                    solar_zenith, zenith_limit):
    # This function is used to fill in nighttime values for the computed
    # irradiance time series (GHI, DHI, DNI).
    # Set the series based on the component.
    if component == 'GHI':
        series = component_sum_df['ghi']
    elif component == 'DHI':
        series = component_sum_df['dhi']
    else:
        series = component_sum_df['dni']
    # Logic for filling in nighttime values for a
    # component sum series.
    # Find the locations where the sun is below the sza limit.
    mask = (zenith_limit <= solar_zenith)
    if isinstance(fill_night_value, float) | isinstance(fill_night_value, int):
        # Replace the nighttime values with a fill value--this can be np.nan,
        # which is a float
        series[mask] = fill_night_value
    elif fill_night_value == 'equation':
        # Use the nighttime equation GHI = 0 + DHI, which translates as:
        # GHI_Calc (at night) = DHI_measured
        # DHI_Calc (at night) = GHI_measured
        # DNI_Calc (at night) = 0
        if component == 'GHI':
            series[mask] = component_sum_df['dhi'][mask]
        elif component == 'DHI':
            series[mask] = component_sum_df['ghi'][mask]
        else:
            series[mask] = 0
    elif fill_night_value is None:
        pass
    else:
        raise ValueError("The fill_night_value variable must be None,"
                         " float or int, or 'equation'. Please change "
                         "the variable fill_night_value value.")
    return series


def calculate_component_sum_series(solar_zenith,
                                   ghi=None,
                                   dhi=None,
                                   dni=None,
                                   dni_clear=None,
                                   zenith_limit=90,
                                   fill_night_value=None):
    r'''
    Use the component sum equations to calculate the missing series, using
    the other available time series. One of the three parameters (ghi, dhi,
    dni) is passed as None, and the two series are used to
    calculate the missing series. After calculation, the series is
    run through a nighttime routine, where nighttime values are set based on
    the fill_night_value parameter.

    The "component sum" or "closure" equation relates the three
    primary irradiance components as follows:

    .. math::

       GHI = DHI + DNI * \cos(\theta_z)


    Parameters
    ----------
    solar_zenith : Series
        Zenith angles in decimal degrees, with datetime index.
        Angles must be >=0 and <=180. Must have the same datetime index
        as `dni`, `dhi`, and `dni`, when available.
    ghi : Series, optional
        Pandas series of GHI data, with datetime index. Must have the same
        datetime index as `dni`, `dhi`, and `solar_zenith`, when available.
    dhi : Series, optional
        Pandas series of DNI data, with datetime index. Must have the same
        datetime index as ghi, dni, and zenith series, when available.
    dni : Series, optional
        Pandas series of dni data, with datetime index. Must have the same
        datetime index as `ghi`, `dhi`, and `solar_zenith`, when available.
    dni_clear : Series, optional
        Pandas series of clearsky dni data. Must have the same datetime index
        as `ghi`, `dhi`, `dni`, and `solar_zenith`, when available. See
        pvlib-python's :py:func:`pvlib.irradiance.dni` for details.
    zenith_limit: Float
        Solar zenith boundary between night and day, in degrees.
        For calculation of the component sum, `solar_zenith` is set to 90 where
        `solar_zenith > zenith_limit`.
    fill_night_value: String or float or int, default None
        Options include 'equation', float or int values (np.nan, 0, etc.), or
        None.
        This is the fill value for nighttime periods.
        If a float or int value is passed (np.nan, 0 , -.5, etc.), then
        nighttime values are filled using the fill_night_value parameter.
        If 'equation' is used, nighttime periods are filled using the
        component sum equation with DNI=0: ``GHI = 0 + DHI``
        If None, then the nighttime values are based on the component sum
        equation.

    Returns
    -------
    Series
        Pandas series of the calculated values, based on the component sum
        equation and corrected for nighttime periods.
    '''  # noqa: E501
    component_sum_df = pvlib.irradiance.complete_irradiance(solar_zenith, ghi,
                                                            dhi, dni,
                                                            dni_clear)
    if ghi is None:
        component = 'GHI'
    elif dhi is None:
        component = 'DHI'
    elif dni is None:
        component = 'DNI'
    else:
        raise ValueError(
            "Please check that exactly one of ghi, dhi and dni parameters "
            "is set to None"
        )
    return _fill_nighttime(component, component_sum_df,
                           fill_night_value,
                           solar_zenith, zenith_limit)
