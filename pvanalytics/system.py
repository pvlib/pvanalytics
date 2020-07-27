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
    rsquared_quadratic = _fit.quadratic(x=envelope.index, y=envelope)
    rsquared_quartic = _fit.quartic_restricted(
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
            quadratic_median = _fit.quadratic(x=median.index, y=median)
            if quadratic_median < median_r2_min:
                return Tracker.UNKNOWN
        elif system_tracking is Tracker.TRACKING:
            quartic_median = _fit.quartic_restricted(
                x=median.index,
                y=median,
                noon=middle
            )
            if quartic_median < median_r2_min:
                return Tracker.UNKNOWN
    return system_tracking


def is_tracking_envelope(series, daytime, clipping, clip_max=0.1,
                         envelope_quantile=0.995, fit_median=True,
                         median_r2_min=0.9, fit_params=None,
                         envelope_min_fraction=0.05,
                         median_min_fraction=0.025,
                         seasonal_split=None):
    """Infer whether the system is equipped with a tracker.

    Data is grouped by minute of the day and a maximum power or
    irradiance envelope (the `envelope_quantile` value at each minute)
    is calculated. Quadratic and quartic curves are fit to this daily
    envelope and the :math:`r^2` of the curve fits are used determine
    whether the system is tracking or fixed.

    If the quadratic fit is a sufficiently good, then
    :py:const:`Tracker.FIXED` is returned.

    If the quartic fit is sufficiently good fit and the quadratic fit
    is sufficiently bad, then :py:const:`Tracker.TRACKING` is
    returned.

    If neither fit is sufficiently good then
    :py:const:`Tracker.UNKNOWN` is returned.

    Optionally, an additional fit is made to the median of the
    data at each minute to confirm the determination of tracking
    or fixed. If performed, tracking or fixed is judged from the fit
    to the median and this result must be consistent with the fit
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
    fit_median : boolean, default True
        Perform a secondary fit with the median power or irradiance to
        validate that the profile is consistent through the entire
        data set.
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
    envelope_min_fraction : float, default 0.05
        After calucluating the power or irradiance envelope data less
        than `envelope_min_fraction` times the maximum of the envelope
        is removed. This excludes data from morning and evening that
        may interfere with curve fitting.
    median_min_fraction : float, default 0.025
        After calculating the median power or irradiance at each
        minute, data less than `median_min_fraction` times the maximum
        is removed. This excludes data from morning and evening that
        may interfere with curve fitting.
    seasonal_split : tuple of list of int, optional
        Tuple specifying a set of winter months and a set of summer
        months. The order is not important. The months are specified
        as integers between 1 and 12. If not specified defaults to
        ([5, 6, 7, 8], [11, 12, 1, 2]) which works well for North
        America. Data is split in to two groups, one for each set and
        curve fits are applied to the upper envelope of each group
        independently. If the curve fits produce different results
        (e.g. one TRACKING and one FIXED) then
        :py:const:`Tracker.UNKNOWN` is returned.

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
    seasonal_split = seasonal_split or ([5, 6, 7, 8], [11, 12, 1, 2])
    series_daytime = series[daytime]
    clip_fraction = (clipping[daytime].sum() / len(clipping[daytime])) * 100
    if clip_fraction > clip_max:
        return Tracker.UNKNOWN
    bounds = _get_bounds(clip_fraction, fit_params)
    first_season = series_daytime[
        series_daytime.index.month.isin(seasonal_split[0])
    ]
    second_season = series_daytime[
        series_daytime.index.month.isin(seasonal_split[1])
    ]
    if len(first_season) == 0 or len(second_season) == 0:
        return _infer_tracking(
            series_daytime,
            envelope_quantile,
            fit_median,
            median_r2_min,
            bounds,
            envelope_min_fraction,
            median_min_fraction
        )
    first_season_tracking = _infer_tracking(
        first_season,
        envelope_quantile,
        fit_median,
        median_r2_min,
        bounds,
        envelope_min_fraction,
        median_min_fraction
    )
    second_season_tracking = _infer_tracking(
        second_season,
        envelope_quantile,
        fit_median,
        median_r2_min,
        bounds,
        envelope_min_fraction,
        median_min_fraction
    )
    if first_season_tracking is not second_season_tracking:
        return Tracker.UNKNOWN
    return first_season_tracking
