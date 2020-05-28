"""Functions for identifying system characteristics."""
import enum
import numpy as np
import scipy.stats
import scipy.optimize


@enum.unique
class Orientation(enum.Enum):
    """Orientation of a PV System can be either Fixed or Tracking."""
    FIXED = 1
    TRACKING = 2


def _remove_morning_evening(data, threshold):
    # Remove morning and evening data by excluding times where
    # `data` is less than ``threshold * data.max()``
    return data[data > threshold * data.max()]


def _group_by_minute(data):
    # return data grouped by minute of the day.
    return data.groupby(data.index.hour * 60 + data.index.minute)


def _fit_quadratic(data):
    # Fit a quadratic to `data` returning R^2 for the fit.
    coefficients = np.polyfit(data.index, data, 2)
    quadratic = np.poly1d(coefficients)
    # Calculate the R^2 for the fit
    _, _, correlation, _, _ = scipy.stats.linregress(
        data, quadratic(data.index)
    )
    return correlation**2


def _fit_quartic(data, middle):
    # Fit a quartic to the data.
    #
    # The quartic must
    # - open downwards
    # - centered within 70 minutes of the middle of `data`
    # - the y-value of the middle must be within 15% of the median of `data`
    #
    # Returns the R^2 for the fit.
    def quartic(x, a, b, c, e):
        return a * (x - e)**4 + b * (x - e)**2 + c
    median = data.median()
    params, _ = scipy.optimize.curve_fit(
        quartic,
        data.index, data,
        bounds=((-1e-05, 0, median * 0.85, middle - 70),
                (-1e-10, median * 3e-05, median * 1.15, middle + 70))
    )
    model = quartic(data.index, params[0], params[1], params[2], params[3])
    residuals = data - model
    return 1 - (np.sum(residuals**2) / np.sum((data - np.mean(data))**2))


def _orientation_from_fit(rsquared_quadratic, rsquared_quartic, clip_percent):
    # Determine orientation based on fit and percent of clipping in the data
    #
    # Returns None if orientation cannot be determined, otherwise
    # returns the orientation
    if clip_percent < 0.5:
        if rsquared_quadratic >= 0.945:
            return Orientation.FIXED
        if rsquared_quartic >= 0.945 and rsquared_quadratic < 0.92:
            return Orientation.TRACKING
    elif clip_percent <= 3.0:
        if rsquared_quadratic >= 0.92:
            return Orientation.FIXED
        if rsquared_quartic >= 0.92 and rsquared_quadratic < 0.92:
            return Orientation.TRACKING
    elif clip_percent <= 4.0:
        if rsquared_quadratic >= 0.90:
            return Orientation.FIXED
        if rsquared_quartic >= 0.92 and rsquared_quadratic < 0.92:
            return Orientation.TRACKING
    elif clip_percent <= 10.0:
        if rsquared_quadratic >= 0.88:
            return Orientation.FIXED
        if rsquared_quartic >= 0.92 and rsquared_quadratic < 0.92:
            return Orientation.TRACKING
    return None


def orientation(series, daytime, clipping, fit_median=True):
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
    fit_median : boolean, default True
        Perform a secondary fit with the median power or irradiance to
        validate that the orientation is consistent through the entire
        data set.

    Returns
    -------
    orientation : Orientation or None
        If the orientation could not be determined returns None,
        otherwise returns the inferred orientation.

    """
    envelope = _remove_morning_evening(
        _group_by_minute(series[daytime]).quantile(0.995),
        0.05
    )
    middle = (envelope.index.max() + envelope.index.min()) / 2
    rsquared_quadratic = _fit_quadratic(envelope)
    rsquared_quartic = _fit_quartic(envelope, middle)
    system_orientation = _orientation_from_fit(
        rsquared_quadratic, rsquared_quartic,
        (clipping[daytime].sum() / len(clipping[daytime])) * 100
    )
    if fit_median:
        median = _remove_morning_evening(
            _group_by_minute(series[daytime]).median(),
            0.025
        )
        if system_orientation is Orientation.FIXED:
            quadratic_median = _fit_quadratic(median)
            if quadratic_median < 0.9:
                return None
        elif system_orientation is Orientation.TRACKING:
            quartic_median = _fit_quartic(median, middle)
            if quartic_median < 0.9:
                return None
    return system_orientation
