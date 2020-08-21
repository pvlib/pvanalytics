"""Functions for normalizing data."""


def min_max(data):
    """Normalize data by its minimum and maximum.

    The following formula is used for normalization

    ..math::
      (x - min(data)) / (max(data) - min(data))

    Parameters
    ----------
    data : Series
        Series of numeric data

    Returns
    -------
    Series

        `data` normalized to the range [0, 1] accoring to the formula
        above.

    """
    minimum = data.min()
    maximum = data.max()
    return (data - minimum) / (maximum - minimum)
