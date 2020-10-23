"""Functions for identifying system characteristics."""
import enum
import warnings
import numpy as np
import scipy
import pandas as pd
import pvlib
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from pvanalytics.util import _fit, _group


@enum.unique
class Tracker(enum.Enum):
    """Enum describing the orientation of a PV System."""

    FIXED = 1
    """A system with a fixed azimuth and tilt."""
    TRACKING = 2
    """A system equipped with a tracker."""
    UNKNOWN = 3
    """A system where the tracking cannot be determined."""


# Default minimum R^2 values for curve fits.
#
# Minimums vary by the fraction of the data that has clipping. Keys
# are tuples with lower (inclusive) and upper (exclusive) bounds for
# the fraction of data with clipping.
PVFLEETS_FIT_PARAMS = {
    (0, 0.005): {'fixed': 0.945, 'tracking': 0.945, 'fixed_max': 0.92},
    (0.005, 0.03): {'fixed': 0.92, 'tracking': 0.92, 'fixed_max': 0.92},
    (0.03, 0.04): {'fixed': 0.9, 'tracking': 0.92, 'fixed_max': 0.92},
    (0.04, 0.1): {'fixed': 0.88, 'tracking': 0.92, 'fixed_max': 0.92},
}


def _remove_morning_evening(data, threshold):
    # Remove morning and evening data by excluding times where
    # `data` is less than ``threshold * data.max()``
    return data[data > threshold * data.max()]


def _is_fixed(rsquared_quadratic, bounds):
    return rsquared_quadratic >= bounds['fixed']


def _is_tracking(rsquared_quartic, rsquared_quadratic, bounds):
    return (
        rsquared_quartic >= bounds['tracking']
        and rsquared_quadratic < bounds['fixed_max']
    )


def _tracking_from_fit(rsquared_quadratic, rsquared_quartic,
                       bounds):
    # Determine if the system has a tracker based on curve fit results.
    #
    # Parameters
    # ----------
    # rsquared_quadratic : float
    #     :math:`r^2` for a quadratic fit.
    # rsquared_quartic : float
    #     :math:`r^2` for a quartic fit.
    # bounds : dictionary
    #     Upper and lower bounds on :math:`r^2` values for a fixed or
    #     tracking system. See values :py:const:`PVFLEETS_FIT_PARAMS` for an
    #     example
    #
    # Returns
    # -------
    # Tracker
    #     Tracking, Fixed, or Unknown.
    if _is_fixed(rsquared_quadratic, bounds):
        return Tracker.FIXED
    if _is_tracking(rsquared_quartic, rsquared_quadratic, bounds):
        return Tracker.TRACKING
    return Tracker.UNKNOWN


def _get_bounds(clip_fraction, fit_params):
    # get the minimum r^2 for fits to determine tracking or fixed. The
    # bounds vary by the fraction of clipping in the data
    # (`clip_fraction`).
    for clip, bounds in fit_params.items():
        if clip[0] <= clip_fraction <= clip[1]:
            return bounds
    return {'tracking': 0.0, 'fixed': 0.0, 'fixed_max': 0.0}


def _infer_tracking(series, envelope_quantile,
                    fit_median, median_r2_min, fit_bounds,
                    envelope_min_fraction, median_min_fraction):
    # Infer system tracking from the upper envelope and median of the
    # data.
    envelope = _remove_morning_evening(
        _group.by_minute(series).quantile(envelope_quantile),
        envelope_min_fraction
    )
    middle = (envelope.index.max() + envelope.index.min()) / 2
    rsquared_quadratic = _fit.quadratic_r2(x=envelope.index, y=envelope)
    rsquared_quartic = _fit.quartic_restricted_r2(
        x=envelope.index,
        y=envelope,
        noon=middle
    )
    system_tracking = _tracking_from_fit(
        rsquared_quadratic, rsquared_quartic,
        fit_bounds
    )
    if fit_median:
        median = _remove_morning_evening(
            _group.by_minute(series).median(),
            median_min_fraction
        )
        if system_tracking is Tracker.FIXED:
            quadratic_median = _fit.quadratic_r2(x=median.index, y=median)
            if quadratic_median < median_r2_min:
                return Tracker.UNKNOWN
        elif system_tracking is Tracker.TRACKING:
            quartic_median = _fit.quartic_restricted_r2(
                x=median.index,
                y=median,
                noon=middle
            )
            if quartic_median < median_r2_min:
                return Tracker.UNKNOWN
    return system_tracking


def _infer_tracking_winter_summer(winter, summer, *args):
    # Infer tracking for winter months and summer months separately.
    #
    # *args should hold all parameters for :py:func:`_infer_tracking`
    # except the data.
    if len(winter) == 0:
        return _infer_tracking(summer, *args)
    if len(summer) == 0:
        return _infer_tracking(winter, *args)
    winter_tracking = _infer_tracking(winter, *args)
    summer_tracking = _infer_tracking(summer, *args)
    if winter_tracking is not summer_tracking:
        return Tracker.UNKNOWN
    return summer_tracking


def is_tracking_envelope(series, daytime, clipping, clip_max=0.1,
                         envelope_quantile=0.995, envelope_min_fraction=0.05,
                         fit_median=True, median_min_fraction=0.025,
                         median_r2_min=0.9, fit_params=None,
                         seasonal_split='north-america'):
    """Infer whether the system is equipped with a tracker.

    Data is grouped by season (optional) and within each season by the
    minute of the day. A maximum power or irradiance envelope (the
    `envelope_quantile` value at each minute) is calculated. Quadratic
    and quartic curves are fit to this daily envelope and the :math:`r^2`
    of the curve fits are used determine whether the system is tracking
    or fixed.

    If the quadratic fit is a sufficiently good in both seasons, then
    :py:const:`Tracker.FIXED` is returned.

    If, in both seasons, the quartic fit is sufficiently good and the
    quadratic fit is sufficiently bad, then :py:const:`Tracker.TRACKING`
    is returned.

    If neither fit is sufficiently good, or the results from each season
    disagree, then :py:const:`Tracker.UNKNOWN` is returned.

    Optionally, an additional fit is made to the median of the
    data at each minute to confirm the determination of tracking
    or fixed. If performed, this result must be consistent with the fit
    to the upper envelope. If not, :py:const:`Tracker.UNKNOWN`
    is returned.

    Parameters
    ----------
    series : Series
        Timezone localized Series of power or irradiance data.
    daytime : Series
        Boolean Series with True for times that are during the day.
    clipping : Series
        Boolean Series identifying where power or irradiance is being
        clipped.
    clip_max : float, default 0.1
        If the fraction of data flagged as clipped is greater than
        `clip_max` then it cannot be determined whether the system is
        tracking or fixed and :py:const:`Tracker.UNKNOWN` is returned.
    envelope_quantile : float, default 0.995
        Quantile used to determine the upper power or irradiance
        envelope.
    envelope_min_fraction : float, default 0.05
        After calculating the power or irradiance envelope, data less
        than `envelope_min_fraction` times the maximum of the envelope
        is removed. This excludes data from morning and evening that
        may interfere with curve fitting.
    fit_median : boolean, default True
        Perform a secondary fit with the median power or irradiance to
        validate that the profile is consistent through the entire
        data set.
    median_min_fraction : float, default 0.025
        After calculating the median power or irradiance at each
        minute, data less than `median_min_fraction` times the maximum
        is removed. This excludes data from morning and evening that
        may interfere with curve fitting.
    median_r2_min : float, default 0.9
        Minimum :math:`r^2` for a curve fit to the median power or
        irradiance at each minute of the day (Applies only if
        `fit_median` is True).
    fit_params : dict or None, default None
        Minimum r-squared for curve fits according to the fraction of
        data with clipping. This should be a dictionary with tuple
        keys and dictionary values. The key must be a 2-tuple of
        ``(clipping_min, clipping_max)`` where the values specify the
        minimum and maximum fraction of data with clipping for which
        the associated fit parameters are applicable. The values of
        the dictionary are themselves dictionaries with keys
        ``'fixed'`` and ``'tracking'``, which give the minimum
        :math:`r^2` for the curve fits, and ``'fixed_max'`` which
        gives the maximum :math:`r^2` for a quadratic fit if the
        system appears to have a tracker.

        If None :py:data:`PVFLEETS_FIT_PARAMS` is used.
    seasonal_split : dict or str or None, default 'north-america'
        A dictionary with two keys, 'winter' and 'summer' with a list of
        integers specifying the winter months and summer months respectively.
        Seasonal grouping can be disabled by passing `seasonal_split=None`.
        Either season can be ignored by passing a dict that omits the key
        or sets its value to None. The default value, 'north-america' uses
        ``{'winter': [11, 12, 1, 2], 'summer': [5, 6, 7, 8]}`` which
        works well for PV systems located in North America.

    Returns
    -------
    Tracker
        The tracking determined by curve fitting (FIXED, TRACKING, or
        UNKNOWN).

    Notes
    -----
    Derived from the PVFleets QA Analysis project.

    See Also
    --------

    pvanalytics.features.orientation.tracking_nrel

    pvanalytics.features.orientation.fixed_nrel

    """
    fit_params = fit_params or PVFLEETS_FIT_PARAMS
    if seasonal_split == 'north-america':
        seasonal_split = {'summer': [5, 6, 7, 8], 'winter': [11, 12, 1, 2]}
    series_daytime = series[daytime]
    clip_fraction = (clipping[daytime].sum() / len(clipping[daytime]))
    if clip_fraction > clip_max:
        return Tracker.UNKNOWN
    bounds = _get_bounds(clip_fraction, fit_params)
    if not seasonal_split:
        return _infer_tracking(
            series_daytime,
            envelope_quantile,
            fit_median,
            median_r2_min,
            bounds,
            envelope_min_fraction,
            median_min_fraction
        )
    summer = series_daytime[
        series_daytime.index.month.isin(seasonal_split.get('summer') or [])
    ]
    winter = series_daytime[
        series_daytime.index.month.isin(seasonal_split.get('winter') or [])
    ]
    if len(winter) == 0 and len(summer) == 0:
        warnings.warn(
            "Unknown tracking - neither winter nor summer months have data."
        )
        return Tracker.UNKNOWN
    return _infer_tracking_winter_summer(
        winter, summer,
        envelope_quantile,
        fit_median,
        median_r2_min,
        bounds,
        envelope_min_fraction,
        median_min_fraction
    )


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


def _power_residuals_from_clearsky(system_params,
                                   ghi, dhi, dni,
                                   power_ac,
                                   solar_zenith, solar_azimuth,
                                   temperature,
                                   wind_speed,
                                   temperature_coefficient,
                                   temperature_model_parameters):
    """Return the residuals between a system with parameters given in
    `system_params` and the data in `power_ac`.

    Parameters
    ----------
    system_params : array-like
        array of four floats: tilt, azimuth, DC capacity, and inverter
        DC input limit.
    ghi : Series
        Clear sky GHI
    dhi : Series
        Clear sky DHI
    dni : Series
        Clear sky DNI
    power_ac : Series
        Measured AC power under clear sky conditions.
    solar_zenith : Series
        Solar zenith at the same times as data in `power_ac`
    solar_azimuth : Series
        Solar azimuth at the same times as data in `power_ac`
    temperature : float or Series
        Air temperature at which to model the hypothetical system. If a
        float then a constant temperature is used. If a Series, must have
        the same index as `power_ac`. [C]
    wind_speed : float or Series
        Wind speed. If a float then a constant wind speed is used. If a
        Series, must have the same index as `power_ac`. [m/s]
    temperature_coefficient : float
        Temperature coefficient of DC power. [1/C]
    temperature_model_parameters : dict
        Parameters for the cell temperature model.

    Returns
    -------
    Series
        Difference between `power_ac` and the PVWatts output with the
        given parameters.

    Notes
    ------
    Uses the defaults in :py:func:`pvlib.irradiance.get_total_irradiance` to
    calculated plane-of-array irradiance, i.e., the isotropic model for sky
    diffuse irradiance, and the Perez irradiance transposition model.
    """
    tilt = system_params[0]
    azimuth = system_params[1]
    dc_capacity = system_params[2]
    dc_inverter_limit = system_params[3]
    poa = pvlib.irradiance.get_total_irradiance(
        tilt, azimuth,
        solar_zenith,
        solar_azimuth,
        dni, ghi, dhi
    )
    temp_cell = pvlib.temperature.sapm_cell(
        poa['poa_global'],
        temperature,
        wind_speed,
        **temperature_model_parameters
    )
    pdc = pvlib.pvsystem.pvwatts_dc(
        poa['poa_global'],
        temp_cell,
        dc_capacity,
        temperature_coefficient
    )
    return power_ac - pvlib.inverter.pvwatts(pdc, dc_inverter_limit)


def _rsquared(data, residuals):
    # Calculate the coefficient of determination from `residuals`
    model = data + residuals
    _, _, r, _, _ = scipy.stats.linregress(model, data)
    return r*r


def infer_orientation_fit_pvwatts(power_ac, ghi, dhi, dni,
                                  solar_zenith, solar_azimuth,
                                  temperature=25, wind_speed=0,
                                  temperature_coefficient=-0.004,
                                  temperature_model_parameters=None):
    """Get the tilt and azimuth that give PVWatts output that most closely
    fits the data in `power_ac`.

    Input data `power_ac`, `ghi`, `dhi`, `dni` should reflect clear-sky
    conditions.

    Uses non-linear least squares to optimize over four free variables
    to find the values that result in the best fit between power modeled
    using PVWatts and `power_ac`. The four free variables are

    - surface tilt
    - surface azimuth
    - the DC capacity of the system
    - the DC input limit of the inverter.

    Of these four parameters, only tilt and azimuth are returned. While, DC
    capacity and the DC input limit are calculated, their values may not be
    accurate. While its value is not returned, because the DC input limit is
    used as a free variable for the optimization process, this function
    can operate on `power_ac` data that includes inverter clipping.

    All parameters passed as a Series must have the same index and must not
    contain any undefined values (i.e. NaNs). If any input contains NaNs a
    ValueError is raised.

    Parameters
    ----------
    power_ac : Series
        AC power from the system in clear sky conditions.
    ghi : Series
        Clear sky GHI with same index as `power_ac`. [W/m^2]
    dhi : Series
        Clear sky DHI with same index as `power_ac`. [W/m^2]
    dni : Series
        Clear sky DNI with same index as `power_ac`. [W/m^2]
    solar_zenith : Series
        Solar zenith. [degrees]
    solar_azimuth : Series
        Solar azimuth. [degrees]
    temperature : float or Series, default 25
        Air temperature at which to model the hypothetical system. If a
        float then a constant temperature is used. If a Series, must have
        the same index as `power_ac`. [C]
    wind_speed : float or Series, default 0
        Wind speed. If a float then a constant wind speed is used. If a
        Series, must have the same index as `power_ac`. [m/s]
    temperature_model_parameters : dict, optional
        Parameters fot the cell temperature model. If not specified
        ``pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm'][
        'open_rack_glass_glass']`` is used. See
        :py:func:`pvlib.temperature.sapm_cell` for more information.

    Returns
    -------
    surface_tilt : float
        Tilt of the array. [degrees]
    surface_azimuth : float
        Azimuth of the array. [degrees]
    r_squared : float
        :math:`r^2` value for the fit at the returned orientation.

    Raises
    ------
    ValueError
        If any input passed as a Series contains undefined values (i.e. NaNs).
    """
    if power_ac.hasnans:
        raise ValueError("power_ac must not contain undefined values")
    if ghi.hasnans or dhi.hasnans or dni.hasnans:
        raise ValueError("ghi, dhi, and dni must not contain undefined values")
    if isinstance(temperature, pd.Series) and temperature.hasnans:
        raise ValueError("temperature must not contain undefined values")
    if isinstance(wind_speed, pd.Series) and wind_speed.hasnans:
        raise ValueError("wind_speed must not contain undefined values")
    if temperature_model_parameters is None:
        temperature_model_parameters = \
            TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    initial_tilt = 45
    initial_azimuth = 180
    initial_dc_capacity = power_ac.max()
    initial_dc_limit = power_ac.max() * 1.5
    fit_result = scipy.optimize.least_squares(
        _power_residuals_from_clearsky,
        [initial_tilt, initial_azimuth, initial_dc_capacity, initial_dc_limit],
        bounds=([0, 0, power_ac.max()*0.5, power_ac.max()*0.5],
                [90, 360, power_ac.max()*2, power_ac.max()*3]),
        kwargs={
            'ghi': ghi,
            'dhi': dhi,
            'dni': dni,
            'solar_zenith': solar_zenith,
            'solar_azimuth': solar_azimuth,
            'power_ac': power_ac,
            'temperature': temperature,
            'temperature_coefficient': temperature_coefficient,
            'wind_speed': wind_speed,
            'temperature_model_parameters': temperature_model_parameters
        }
    )
    r_squared = _rsquared(power_ac, fit_result.fun)
    return fit_result.x[0], fit_result.x[1], r_squared
