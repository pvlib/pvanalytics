"""General purpose utility functions."""
import pandas as pd
from pandas.tseries import frequencies


def freq_to_timedelta(freq):
    """Convert a pandas freqstr to a timedelta

    Parameters
    ----------
    freq : str

    Returns
    -------
    Timedelta
    """
    return pd.to_timedelta(frequencies.to_offset(freq))
