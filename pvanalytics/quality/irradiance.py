"""Quality control functions for irradiance data."""

import numpy as np
from pvlib.tools import cosd
import pvlib

from pvanalytics.quality import util


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

    """
    if not limits:
        limits = QCRAD_LIMITS
    ghi_ub = _qcrad_ub(dni_extra, solar_zenith, limits['ghi_ub'])

    ghi_limit_flag = util.check_limits(ghi, limits['ghi_lb'], ghi_ub)

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

    """
    if not limits:
        limits = QCRAD_LIMITS

    dhi_ub = _qcrad_ub(dni_extra, solar_zenith, limits['dhi_ub'])

    dhi_limit_flag = util.check_limits(dhi, limits['dhi_lb'], dhi_ub)

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

    """
    if not limits:
        limits = QCRAD_LIMITS

    dni_ub = _qcrad_ub(dni_extra, solar_zenith, limits['dni_ub'])

    dni_limit_flag = util.check_limits(dni, limits['dni_lb'], dni_ub)

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
    dhi_limit_flag : Series
        True for each value that is physically possible. None if `dhi` is None.

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
        util.check_limits(sza, lower_bound=sza_lb,
                          upper_bound=sza_ub, inclusive_lower=True)
        & util.check_limits(ghi, lower_bound=ghi_lb, upper_bound=ghi_ub)
        & util.check_limits(ratio, lower_bound=ratio_lb, upper_bound=ratio_ub)
    )


def check_irradiance_consistency_qcrad(ghi, solar_zenith, dhi, dni,
                                       param=None):
    """Check consistency of GHI, DHI and DNI using QCRad criteria.

    Uses criteria given in [1]_ to validate the ratio of irradiance
    components.

    .. warning:: Not valid for night time. While you can pass data
       from night time to this function, be aware that the truth
       values returned for that data will not be valid.

    Parameters
    ----------
    ghi : Series
        Global horizontal irradiance in :math:`W/m^2`
    solar_zenith : Series
        Solar zenith angle in degrees
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


def ghi_clearsky_limits(ghi, ghi_clearsky, csi_max=1.1):
    """Identify GHI values greater than clearsky values.

    Uses :py:func:`pvlib.irradiance.clearsky_index` to compute the
    clearsky index for `ghi` and `ghi_clearsky`. Compares the
    clearsky index to `csi_max` to identify values in `ghi` that are
    greater than the expected clearsky irradiance.

    Parameters
    ----------
    ghi : Series
        Global horizontal irradiance in :math:`W/m^2`
    ghi_clearsky : Series
        Global horizontal irradiance in :math:`W/m^2` under clearsky
        conditions.
    csi_max : float, default 1.1
        Maximum acceptable clearsky index. Any clearsky index value
        greater than this indicates a `ghi` value that is too large.

    Returns
    -------
    Series
        True for each value where the clearsky index does not exceed
        `csi_max`.

    """
    csi = pvlib.irradiance.clearsky_index(
        ghi,
        ghi_clearsky,
        max_clearsky_index=np.Inf
    )
    return util.check_limits(csi, upper_bound=csi_max, inclusive_upper=True)


def poa_clearsky_limits(poa, poa_clearsky, csi_max=1.1):
    """Identify POA irradiance values greater than clearsky values.

    Uses :py:func:`pvlib.irradiance.clearsky_index` to compute the
    clearsky index for `poa` and `poa_clearsky`. Compares the
    clearsky index to `csi_max` to identify values in `poa` that are
    greater than the expected clearsky irradiance.

    Parameters
    ----------
    poa : Series
        Plane of array irradiance in :math:`W/m^2`
    poa_clearsky : Series
        Plane of array irradiance in :math:`W/m^2` under clearsky
        conditions.
    csi_max : float, default 1.1
        Maximum clearsky index that defines when POA irradiance exceeds
        clear-sky value.

    Returns
    -------
    Series
        True for each value where the clearsky index does not exceed
        `csi_max`.

    """
    csi = pvlib.irradiance.clearsky_index(
        poa,
        poa_clearsky,
        max_clearsky_index=np.Inf
    )
    return util.check_limits(csi, upper_bound=csi_max, inclusive_upper=True)
