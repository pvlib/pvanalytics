"""Functions for identifying system characteristics."""
from enum import Enum


class Orientation(Enum):
    """Orientation of a PV System can be either Fixed or Tracking."""
    FIXED = auto()
    TRACKING = auto()


def orientation(power, daylight, clipping=None):
    """Infer the orientation of the system from power data.

    Parameters
    ----------
    power : Series
        Time series of AC power data.
    daylight : Series
        Boolean Series with True for times that are during the day.
    clipping : Series, default None
        Boolean Series identifying where power is being clipped.

    Returns
    -------
    Orientation or None
        If the orientation could not be determined returns None,
        otherwise returns the inferred orientation.

    """
    # group data by minute and compute the 99.5% quantile.
    daytime_power = power[daylight]
    power_by_minute = daytime_power.groupby(
        power.index.hour * 60 + power.index.minute
    )
    by_minute_quantile = power_by_minute.quantile(0.995)
    by_minute_median = power_by_minute.median()
    # remove low values (these would be morning and evening times)
    upper_envelope = by_minute_quantile[
        by_minute_quantile > 0.05 * by_minute_quantile.max()
    ]
    median_envelope = by_minute_median[
        by_minute_median > 0.025 * by_minute_median.max()
    ]
    quantile_middle = (upper_envelope.index.max()
                       + upper_envelope.index.min()) / 2
    # try different curve fitting to identify the orientation.
    # TODO
    return None
