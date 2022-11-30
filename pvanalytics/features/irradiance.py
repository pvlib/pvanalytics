"""Quality control functions for irradiance data."""

import numpy as np
import datetime as dt


def nighttime_offset_correction(irradiance, zenith, sza_night_limit=100,
                                label='right', midnight_method='zenith',
                                aggregation_method='median'):
    """
    Apply nighttime correction to irradiance time series.

    Parameters
    ----------
    irradiance : pd.Series
        Pandas Series of irradiance data.
    zenith : pd.Series
        Pandas Series of zenith angles corresponding to the irradiance time
        series.
    sza_night_limit : float, optional
        Solar zenith angle boundary limit (periods with zenith angles greater
        or equal are used to compute the nighttime offset). The default is 100.
    label : {'right', 'left'}, optional
        Whether the timestamps correspond to the start/left or end/right of the
        interval. The default is 'right'.
    midnight_method : {'zenith', 'time'}, optional
        Method for determining midnight. The default is 'zenith', which
        assumes midnight occurs when the zenith angle is at the maximum.
    aggregation_method : {'median', 'mean'}, optional
        Method for calculating nighttime offset. The default is 'median'.

    Returns
    -------
    corrected_irradiance : pd.Series
        Pandas Series of nighttime corrected irradiance.
    """
    # Raise an error if arguments are incorrect
    if label not in ['right', 'left']:
        raise ValueError("label must be 'right' or 'left'.")
    if aggregation_method not in ['median', 'mean']:
        raise ValueError("aggregation_method must be 'mean' or 'median'.")

    # Create boolean series where nighttime is one (calculated based on the
    # zenith angle)
    midnight_zenith = (zenith.diff().apply(np.sign).diff() < 0)
    # Assign unique number to each day
    day_number_zenith = midnight_zenith.cumsum()

    # Choose grouping parameter based on the midnight_method
    if midnight_method == 'zenith':
        grouping_category = day_number_zenith
    elif midnight_method == 'time':
        grouping_category = irradiance.index.date
        if label == 'right':
            grouping_category[irradiance.index.time == dt.time(0)] += -dt.timedelta(days=1)
    else:
        raise ValueError("midnight_method must be 'zenith' or 'time'.")

    # Create Pandas Series only containing nighttime irradiance
    nighttime_irradiance = irradiance[zenith >= sza_night_limit]
    # Calculate nighttime offset
    nighttime_offset = nighttime_irradiance.groupby(grouping_category).transform(aggregation_method)
    # Calculate corrected irradiance time series
    corrected_irradiance = irradiance - nighttime_offset
    return corrected_irradiance
