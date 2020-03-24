"""Quality control functions for identifying gaps in the data.

Gaps include missing data, interpolation, stuck values, and filler
values (i.e. 999).

"""
import numpy as np


def _all_close_to_first(x, rtol=1e-5, atol=1e-8):
    """Test if all values in x are close to x[0].

    Parameters
    ----------
    x : array
    rtol : float, default 1e-5
        Relative tolerance for detecting a change in data values.
    atol : float, default 1e-8
        Absolute tolerance for detecting a change in data values.

    Returns
    -------
    Boolean

    """
    return np.allclose(a=x, b=x[0], rtol=rtol, atol=atol)


def detect_stale_values(x, window=3, rtol=1e-5, atol=1e-8):
    """Identify stale values in the data.

    For a window of length N, the last value (index N-1) is considered
    stale if all values in the window are close to the first value
    (index 0).

    Parameters
    ----------
    x : Series
        data to be processed
    window : int, default 3
        number of consecutive values which, if unchanged, indicates
        stale data
    rtol : float, default 1e-5
        relative tolerance for detecting a change in data values
    atol : float, default 1e-8
        absolute tolerance for detecting a change in data values

    Parameters rtol and atol have the same meaning as in
    numpy.allclose

    Returns
    -------
    Series
        True for each value that is part of a stale sequence of data

    Raises
    ------
    ValueError
        If window < 2.

    """
    if window < 2:
        raise ValueError('window set to {}, must be at least 2'.format(window))

    flags = x.rolling(window=window).apply(
        _all_close_to_first,
        raw=True,
        kwargs={'rtol': rtol, 'atol': atol}
    ).fillna(False).astype(bool)
    return flags


def detect_interpolation(x, window=3, rtol=1e-5, atol=1e-8):
    """Identify sequences which appear to be linear.

    Sequences are linear if the first difference appears to be
    constant.  For a window of length N, the last value (index N-1) is
    flagged if all values in the window appear to be a line segment.

    Parameters
    ----------
    x : Series
        data to be processed
    window : int, default 3
        number of sequential values that, if the first difference is
        constant, are classified as a linear sequence
    rtol : float, default 1e-5
        tolerance relative to max(abs(x.diff()) for detecting a change
    atol : float, default 1e-8
        absolute tolerance for detecting a change in first difference

    Returns
    -------
    Series
        True for each value that is part of a linear sequence

    Raises
    ------
    ValueError
        If window < 3.

    """
    if window < 3:
        raise ValueError('window set to {}, must be at least 3'.format(window))

    # reduce window by 1 because we're passing the first difference
    flags = detect_stale_values(
        x.diff(periods=1),
        window=window-1,
        rtol=rtol,
        atol=atol
    )
    return flags
