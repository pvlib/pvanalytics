"""Common fixtures for features tests."""
import pytest
import numpy as np
import pandas as pd
from pvlib import location


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def quadratic():
    """Downward facing quadratic.

    Vertex at index 30 and roots at indices 0, 60.

    """
    q = -1000 * (np.linspace(-1, 1, 61) ** 2) + 1000
    return pd.Series(q)


@pytest.fixture(scope='module')
def one_year_hourly():
    return pd.date_range(
        start='03/01/2020',
        end='03/01/2021',
        closed='left',
        freq='H',
        tz='Etc/GMT+7'
    )


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


@pytest.fixture(scope='module')
def clearsky_year(one_year_hourly, albuquerque):
    """One year of hourly clearsky data."""
    return albuquerque.get_clearsky(
        one_year_hourly,
        model='simplified_solis'
    )


@pytest.fixture(scope='module')
def solarposition_year(one_year_hourly, albuquerque):
    """One year of solar position data in albuquerque"""
    return albuquerque.get_solarposition(one_year_hourly)
