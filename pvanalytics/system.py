"""Functions for identifying system characteristics."""
import enum
from pvanalytics.util import _fit, _group


@enum.unique
class Orientation(enum.Enum):
    """Orientation of a PV System can be either Fixed or Tracking."""
    FIXED = 1
    TRACKING = 2
    UNKNOWN = 3


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


def _orientation_from_fit(rsquared_quadratic, rsquared_quartic,
                          clip_percent, clip_max, fit_params):
    # Determine orientation based on fit and percent of clipping in the data
    #
    # Returns Orientation.UNKNOWN if orientation cannot be determined,
    # otherwise returns the orientation.
    if clip_percent > clip_max:
        # Too much clipping means the orientation cannot be determined
        return Orientation.UNKNOWN
    bounds = _get_bounds(clip_percent, fit_params)
    if _is_fixed(rsquared_quadratic, bounds):
        return Orientation.FIXED
    if _is_tracking(rsquared_quartic, rsquared_quadratic, bounds):
        return Orientation.TRACKING
    return Orientation.UNKNOWN


def _get_bounds(clip_percent, fit_params):
    # get the minimum r^2 for fits to determine tracking or fixed
    # orientation. The bounds vary by the amount of clipping in the
    # data, passed as a percentage in `clip_percent`.
    for clip, bounds in fit_params.items():
        if clip[0] <= clip_percent <= clip[1]:
            return bounds
    return {'tracking': 0.0, 'fixed': 0.0, 'fixed_max': 0.0}


def orientation(series, daytime, clipping, clip_max=10.0,
                fit_median=True, fit_params=None):
    """Infer the orientation of the system from power or irradiance data.

    Parameters
    ----------
    series : Series
        Time series of power or irradiance data.
    daytime : Series
        Boolean Series with True for times that are during the day.
    clipping : Series
        Boolean Series identifying where power or irradiance is being
        clipped.
    clip_max : float, default 10.0
        If the percent of data flagged as clipped is greater than
        `clip_max` then the orientation cannot be determined and
        :py:const:`Orientation.UNKNOWN` is returned.
    fit_median : boolean, default True
        Perform a secondary fit with the median power or irradiance to
        validate that the orientation is consistent through the entire
        data set.
    fit_params : dict or None, default None
        Minimum r-squared for curve fits according to the fraction of
        data with clipping. If None :py:data:`PVFLEETS_FIT_PARAMS` is
        used.

    Returns
    -------
    Orientation
        The orientation determined by curve fitting.

    """
    fit_params = fit_params or PVFLEETS_FIT_PARAMS
    envelope = _remove_morning_evening(
        _group.by_minute(series[daytime]).quantile(0.995),
        0.05
    )
    middle = (envelope.index.max() + envelope.index.min()) / 2
    rsquared_quadratic = _fit.quadratic(envelope)
    rsquared_quartic = _fit.quartic_restricted(envelope, middle)
    system_orientation = _orientation_from_fit(
        rsquared_quadratic, rsquared_quartic,
        (clipping[daytime].sum() / len(clipping[daytime])) * 100,
        clip_max,
        fit_params
    )
    if fit_median:
        median = _remove_morning_evening(
            _group.by_minute(series[daytime]).median(),
            0.025
        )
        if system_orientation is Orientation.FIXED:
            quadratic_median = _fit.quadratic(median)
            if quadratic_median < 0.9:
                return Orientation.UNKNOWN
        elif system_orientation is Orientation.TRACKING:
            quartic_median = _fit.quartic_restricted(median, middle)
            if quartic_median < 0.9:
                return Orientation.UNKNOWN
    return system_orientation
