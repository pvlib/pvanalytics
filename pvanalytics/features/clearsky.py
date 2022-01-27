"""Functions for identifying periods of clear sky conditions."""
import numpy as np
import pvlib


def reno(ghi, ghi_clearsky):
    """Identify times when GHI is consistent with clearsky conditions.

    Uses the function :py:func:`pvlib.clearsky.detect_clearsky`.

    .. note::

       Must be given GHI data with regular (constant) time intervals
       of 15 minutes or less.

    Parameters
    ----------
    ghi : Series
        Global horizontal irradiance in :math:`W/m^2`. Must have an
        index with time intervals of at most 15 minutes.
    ghi_clearsky : Series
        Global horizontal irradiance in :math:`W/m^2` under clearsky
        conditions.

    Returns
    -------
    Series
        True when clear sky conditions are indicated.

    Raises
    ------
    ValueError
        if the time intervals are greater than 15 minutes.

    Notes
    -----
    Clear-sky conditions are inferred when each of six criteria are
    met; see :py:func:`pvlib.clearsky.detect_clearsky` for references
    and details. Threshold values for each criterion were originally
    developed for ten minute windows containing one-minute data
    [1]_. As indicated in [2]_, the algorithm also works for longer
    windows and data at different intervals if threshold criteria are
    roughly scaled to the window length. Here, the threshold values
    are based on [1] with the scaling indicated in [2].

    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    References
    ----------
    .. [1] Reno, M.J. and C.W. Hansen, "Identification of periods of
       clear sky irradiance in time series of GHI measurements"
       Renewable Energy, v90, p. 520-531, 2016.

    .. [2] B. H. Ellis, M. Deceglie and A. Jain, "Automatic Detection
       of Clear-Sky Periods From Irradiance Data," in IEEE Journal of
       Photovoltaics, vol. 9, no. 4, pp. 998-1005, July 2019. doi:
       10.1109/JPHOTOV.2019.2914444

    """
    delta = ghi.index.to_series().diff()
    delta_minutes = delta[1].total_seconds() / 60
    if delta_minutes > 15:
        raise ValueError('clearsky requires regular time intervals '
                         'of 15m or less')
    window_length = np.minimum(10*delta_minutes, 60.0)
    scale_factor = window_length / 10
    flags = pvlib.clearsky.detect_clearsky(
        ghi,
        ghi_clearsky,
        ghi.index,
        window_length,
        lower_line_length=-5*scale_factor,
        upper_line_length=10*scale_factor,
        slope_dev=8*scale_factor
    )
    return flags
