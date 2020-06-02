"""Internal module for curve fitting functions."""
import numpy as np
import scipy.optimize


def quadratic(data):
    """Fit a quadratic to the data.

    Parameters
    ----------
    data : Series
        Series of power or irradiance measurements.

    Reurns
    ------
    rsquared : float
        The :math:`R^2` value for the fit.
    coefficients : list(float)
        The coefficients of the quadratic.
    """
    # Fit a quadratic to `data` returning R^2 for the fit.
    coefficients = np.polyfit(data.index, data, 2)
    quadratic = np.poly1d(coefficients)
    # Calculate the R^2 for the fit
    _, _, correlation, _, _ = scipy.stats.linregress(
        data, quadratic(data.index)
    )
    return correlation**2, coefficients


def quartic(data, noon=None):
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
    data : Series
        power or irradiance data.
    noon : int or None, default None
       The index of solar noon. If None then

    Returns
    -------
    rsquared : float
        The :math:`R^2` value for the fit.

    """
    def _quartic(x, a, b, c, e):
        return a * (x - e)**4 + b * (x - e)**2 + c
    median = data.median()
    params, _ = scipy.optimize.curve_fit(
        _quartic,
        data.index, data,
        bounds=((-1e-05, 0, median * 0.85, noon - 70),
                (-1e-10, median * 3e-05, median * 1.15, noon + 70))
    )
    model = _quartic(data.index, params[0], params[1], params[2], params[3])
    residuals = data - model
    return 1 - (np.sum(residuals**2) / np.sum((data - np.mean(data))**2))
