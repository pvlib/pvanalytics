"""Internal module for curve fitting functions."""
import numpy as np
import scipy.optimize


def _quadratic(xs, ys):
    # fit a quadratic function of `xs` to the data in `ys`
    coefficients = np.polyfit(xs, ys, 2)
    return np.poly1d(coefficients)


def quadratic_idxmax(x, y, model_range=None):
    """Fit a quadratic to the x, y data and return the x-value of the vertex.

    Parameters
    ----------
    x : array_like
    y : Series
        x and y-values to fit to.
    model_range : array_like, optional
        Range of x-values to find the max over. If not specified then
        `x` is used.

    Returns
    -------
    numeric
        x-value of the vertex of a quadratic fit to the data in `x`
        and `y`.

    Examples
    --------
    >>> quadratic_idxmax(x=[-2, -1, 2, 4], y=[-4, -1, -4, -16])
    -1

    >>> quadratic_idxmax(
    ...     x=[-2, -1, 2, 4],
    ...     y=[-4, -1, -4, -16],
    ...     model_range=[-4, -3, -2, -1, 0, 1, 2, 3, 4]
    ... )
    0

    """
    model_range = model_range or x
    quadratic = _quadratic(x, y)
    model = quadratic(model_range)
    return model_range[np.argmax(model)]


def quadratic(x, y):
    """Fit a quadratic to the data.

    Parameters
    ----------
    x : array_like
        x values for data in `y`.
    y : Series
        data to which the curve should be fit.

    Reurns
    ------
    float
        The :math:`R^2` value for the fit.

    Examples
    --------
    This function is typically used for fitting a function to power or
    irradiance data. In this case the irradiance measurements are
    passed as `y` and the time of day for each value, in minutes since
    midnight, is passed as `x`. Suppose ``ghi`` below is a time series
    with one day of GHI measurements:

    >>> r2 = quadratic(
    ...     y=ghi
    ...     x=ghi.index.minute + ghi.index.hour * 60,
    ... )

    Notes
    -----
    Based on the PVFleets QA Analysis project. Copyright (c) 2020
    Alliance for Sustainable Energy, LLC.

    """
    quadratic = _quadratic(x, y)
    _, _, correlation, _, _ = scipy.stats.linregress(
        y, quadratic(x)
    )
    return correlation**2


def quartic_restricted(x, y, noon=720):
    """Fit a restricted quartic to the data.

    The quartic is restricted to match the expected shape for a
    tracking pv system under clearsky conditions. The quartic must:

    - open downwards
    - be centered within 70 minutes of the expected solar noon (see
      `noon` parameter)
    - the y-value at the center must be within 15% of the median of
      `data`

    Parameters
    ----------
    x : array_like
        x values for data in `y`
    y : Series
        values to which the curve should be fit
    noon : int, default 720
       The minute for solar noon. Defaults to the clock-noon.

    Returns
    -------
    rsquared : float
        The :math:`R^2` value for the fit.

    Examples
    --------
    This function is typically used for fitting a function to power or
    irradiance data. In this case the irradiance measurements are
    passed as `y` and the time of day for each value, in minutes since
    midnight, is passed as `x`. Suppose ``poa`` below is a time series
    with one day of POA irradiance measurements:

    >>> r2 = quartic_restricted(
    ...     y=poa
    ...     x=poa.index.minute + poa.index.hour * 60,
    ... )

    Notes
    -----
    Based on the PVFleets QA Analysis project. Copyright (c) 2020
    Alliance for Sustainable Energy, LLC.

    """
    def _quartic(x, a, b, c, e):
        return a * (x - e)**4 + b * (x - e)**2 + c
    median = y.median()
    params, _ = scipy.optimize.curve_fit(
        _quartic,
        x, y,
        bounds=((-1e-05, 0, median * 0.85, noon - 70),
                (-1e-10, median * 3e-05, median * 1.15, noon + 70))
    )
    model = _quartic(x, params[0], params[1], params[2], params[3])
    residuals = y - model
    return 1 - (np.sum(residuals**2) / np.sum((y - np.mean(y))**2))
