"""Internal module for curve fitting functions."""
import numpy as np
import scipy.optimize


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
    coefficients = np.polyfit(x, y, 2)
    quadratic = np.poly1d(coefficients)
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
