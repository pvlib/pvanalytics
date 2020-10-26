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
    """
    return pd.to_timedelta(frequencies.to_offset(freq))
