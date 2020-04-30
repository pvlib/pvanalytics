"""Common fixtures for features tests."""
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def quadratic():
    """Downward facing quadratic.

    Vertex at index 30 and roots at indices 0, 60.

    """
    q = -1000 * (np.linspace(-1, 1, 61) ** 2) + 1000
    return pd.Series(q)


@pytest.fixture
def quadratic_day(quadratic):
    """Same as `quadratic`, but embedded in a 24 hour time series.

    The quadratic begins at midnight January 1, 2020. The series has a
    ten-minute frequency (so the quadratic ends 610 minutes after
    midnight). All other values are set to zero.

    """
    day = quadratic.copy()
    day.index = pd.date_range(
        start='1/1/2020 00:00', freq='10T', periods=len(quadratic)
    )
    return day.reindex(
        pd.date_range(start='1/1/2020 00:00',
                      end='1/1/2020 23:50',
                      freq='10T'),
        fill_value=0
    )
