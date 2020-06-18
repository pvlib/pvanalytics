"""Common fixtures for features tests."""
import pytest
import numpy as np
import pandas as pd
from pvlib import location


@pytest.fixture
def quadratic():
    """Downward facing quadratic.

    Vertex at index 30 and roots at indices 0, 60.

    """
    q = -1000 * (np.linspace(-1, 1, 61) ** 2) + 1000
    return pd.Series(q)


@pytest.fixture(scope='module')
def three_days_hourly():
    """Three days with one hour timestamp spacing in Etc/GMT+7"""
    return pd.date_range(
        start='03/01/2020',
        end='03/04/2020',
        closed='left',
        freq='H',
        tz='Etc/GMT+7'
    )


@pytest.fixture(scope='module')
def albuquerque():
    """pvlib Location for Albuquerque, NM."""
    return location.Location(
        35.0844,
        -106.6504,
        name='Albuquerque',
        altitude=1500,
        tx='Etc/GMT+7'
    )


@pytest.fixture(scope='module')
def clearsky(three_days_hourly, albuquerque):
    """Clearsky at `three_days_hourly` in `albuquerque`."""
    return albuquerque.get_clearsky(
        three_days_hourly,
        model='simplified_solis'
    )


@pytest.fixture(scope='module')
def solarposition(three_days_hourly, albuquerque):
    """Solar position at `three_days_hourly` in `albuquerque`."""
    return albuquerque.get_solarposition(three_days_hourly)
