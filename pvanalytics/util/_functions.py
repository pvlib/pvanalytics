"""General purpose utility functions."""
import pandas as pd
from pandas.tseries import frequencies


def freq_to_timedelta(freq):
    """Convert a pandas freqstr to a Timedelta.

    Parameters
    ----------
    freq : str
        A pandas freqstr (e.g. '10T').
    Returns
    -------
    Timedelta
        Timedelta corresponding to a single period at `freq`.
        For offsets that cannot be directly converted (e.g. 'D', 'W'),
        falls back to using the offset's nanosecond representation.
    """
    offset = frequencies.to_offset(freq)
    try:
        return pd.to_timedelta(offset)
    except ValueError:
        return pd.Timedelta(offset.nanos, unit='ns')
