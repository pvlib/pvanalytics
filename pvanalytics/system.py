"""Functions for identifying system characteristics."""
import enum
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
# the clipping percent.
PVFLEETS_FIT_PARAMS = {
    (0, 0.5): {'fixed': 0.945, 'tracking': 0.945, 'fixed_max': 0.92},
    (0.5, 3): {'fixed': 0.92, 'tracking': 0.92, 'fixed_max': 0.92},
    (3, 4): {'fixed': 0.9, 'tracking': 0.92, 'fixed_max': 0.92},
    (4, 10): {'fixed': 0.88, 'tracking': 0.92, 'fixed_max': 0.92},
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
                       clip_percent, clip_max, fit_params):
    # Determine tracking based on fit and percent of clipping in the data.
    #
    # If `clip_percent` is greater than `clip_max` or the curve fits
    # are not sufficiently good then Tracker.UNKNOWN is
    # returned. Otherwise Tracker.FIXED or Tracker.TRACKING is
    # returned.
    if clip_percent > clip_max:
        # Too much clipping means the tracking cannot be determined
        return Tracker.UNKNOWN
    bounds = _get_bounds(clip_percent, fit_params)
    if _is_fixed(rsquared_quadratic, bounds):
        return Tracker.FIXED
    if _is_tracking(rsquared_quartic, rsquared_quadratic, bounds):
        return Tracker.TRACKING
    return Tracker.UNKNOWN


def _get_bounds(clip_percent, fit_params):
    # get the minimum r^2 for fits to determine tracking or fixed. The
    # bounds vary by the amount of clipping in the data, passed as a
    # percentage in `clip_percent`.
    for clip, bounds in fit_params.items():
        if clip[0] <= clip_percent <= clip[1]:
            return bounds
    return {'tracking': 0.0, 'fixed': 0.0, 'fixed_max': 0.0}


def _infer_tracking(series, clip_percent, clip_max, envelope_quantile,
                    fit_median, fit_params):
    # Infer system tracking from the upper envelope and median of the
    # data.
    envelope = _remove_morning_evening(
        _group.by_minute(series).quantile(envelope_quantile),
        0.05
    )
    middle = (envelope.index.max() + envelope.index.min()) / 2
    rsquared_quadratic = _fit.quadratic(envelope)
    rsquared_quartic = _fit.quartic_restricted(envelope, middle)
    system_tracking = _tracking_from_fit(
        rsquared_quadratic, rsquared_quartic,
        clip_percent,
        clip_max,
        fit_params
    )
    if fit_median:
        median = _remove_morning_evening(
            _group.by_minute(series).median(),
            0.025
        )
        if system_tracking is Tracker.FIXED:
            quadratic_median = _fit.quadratic(median)
            if quadratic_median < 0.9:
                return Tracker.UNKNOWN
        elif system_tracking is Tracker.TRACKING:
            quartic_median = _fit.quartic_restricted(median, middle)
            if quartic_median < 0.9:
                return Tracker.UNKNOWN
    return system_tracking


def is_tracking_envelope(series, daytime, clipping, clip_max=10.0,
                         envelope_quantile=0.995, fit_median=True,
                         fit_params=None):
    """Infer whether the system is equipped with a tracker.

    Data is grouped by minute of the day and a maximum power or
    irradiance envelope (the `envelope_quantile` value at each minute)
    is calculated. Quadratic and quartic curves are fit to this daily
    envelope and the :math:`r^2` of the curve fits are used determine
    whether the system is tracking or fixed. Finally an additional fit
    is performed on the median of the data at each minute to determine
    if there is substantial data below the envelope that does not
    match the same profile. If the quadratic has a sufficiently good
    fit then :py:const:`Tracker.FIXED` is returned. If the quartic has
    a sufficiently good fit and the quadratic has a sufficiently bad
    fit then :py:const:`Tracker.TRACKING` is returned. If neiher curve
    fits well or there is a mismatch between the fit to the upper
    envelope and the fit to the median then
    :py:const:`Tracker.UNKNOWN` is returned.

    Parameters
    ----------
    series : Series
        Timezone localized Series of power or irradiance data.
    daytime : Series
        Boolean Series with True for times that are during the day.
    clipping : Series
        Boolean Series identifying where power or irradiance is being
        clipped.
    clip_max : float, default 10.0
        If the percent of data flagged as clipped is greater than
        `clip_max` then tracking cannot be determined and
        :py:const:`Tracker.UNKNOWN` is returned.
    envelope_quantile : float, default 0.995
        Quantile used to determine the upper power or irradiance
        envelope.
    fit_median : boolean, default True
        Perform a secondary fit with the median power or irradiance to
        validate that the profile is consistent through the entire
        data set.
    fit_params : dict or None, default None
        Minimum r-squared for curve fits according to the fraction of
        data with clipping. This should be a dictionary with tuple
        keys and dictionary values. The key must be a 2-tuple of
        ``(clipping_min, clipping_max)`` where the values specify the
        minimum and maximum percent of data with clipping for which
        the associated fit parameters are applicable. The values of
        the dicationary are themselves dictionaries with keys
        ``'fixed'`` and ``'tracking'``, which give the minimum
        :math:`r^2` for the curve fits, and ``'fixed_max'`` which
        gives the maximum :math:`r^2` for a quadratic fit if the
        system appears to have a tracker.

        If None :py:data:`PVFLEETS_FIT_PARAMS` is used.

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
    series_daytime = series[daytime]
    clip_percent = (clipping[daytime].sum() / len(clipping[daytime])) * 100
    march_september = series_daytime[
        series_daytime.index.month.isin([5, 6, 7, 8])
    ]
    october_february = series_daytime[
        series_daytime.index.month.isin([11, 12, 1, 2])
    ]
    if len(october_february) == 0 or len(march_september) == 0:
        return _infer_tracking(
            series_daytime,
            clip_percent,
            clip_max,
            envelope_quantile,
            fit_median,
            fit_params
        )
    march_september_tracking = _infer_tracking(
        march_september,
        clip_percent,
        clip_max,
        envelope_quantile,
        fit_median,
        fit_params
    )
    october_february_tracking = _infer_tracking(
        october_february,
        clip_percent,
        clip_max,
        envelope_quantile,
        fit_median,
        fit_params
    )
    if march_september_tracking is not october_february_tracking:
        return Tracker.UNKNOWN
    return march_september_tracking
