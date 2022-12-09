"""Common fixtures for features tests."""
import pytest
import numpy as np
import pandas as pd
import pvlib
from pvlib import location, pvsystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from pathlib import Path
from pkg_resources import Requirement, parse_version

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR.parent / 'data'


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "pdc0_inverter: pass inverter"
                                       "DC input limit to fixture that"
                                       "models AC power using PVWatts")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def requires_pvlib(versionspec, reason=''):
    """
    Decorator to skip pytest tests if a dependency version is not satisfied.

    Parameters
    ----------
    versionspec : str
        A version specifier like '==0.8.1' or '<=0.9.0'
    reason : str, optional
        Additional context to show in pytest log output
    """
    req = Requirement.parse('pvlib' + versionspec)
    is_satisfied = parse_version(pvlib.__version__) in req
    message = 'requires pvlib' + versionspec
    if reason:
        message += f'({reason})'
    return pytest.mark.skipif(not is_satisfied, reason=message)


def requires_ruptures(test):
    """Skip `test` if ruptures is not installed."""
    try:
        import ruptures  # noqa: F401
        has_ruptures = True
    except ImportError:
        has_ruptures = False
    return pytest.mark.skipif(
        not has_ruptures, reason="requires ruptures")(test)


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
        tz='Etc/GMT+7'
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


@pytest.fixture(scope='module')
def array_parameters():
    """Array parameters for generating simulated power data."""
    sandia_modules = pvsystem.retrieve_sam('SandiaMod')
    module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
    temperature_model_parameters = (
        TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    )
    return {
        'module_parameters': module,
        'temperature_model_parameters': temperature_model_parameters
    }


@pytest.fixture(scope='module')
def system_parameters():
    """PVSystem parameters for generating simulated power data."""
    sapm_inverters = pvsystem.retrieve_sam('cecinverter')
    inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
    return {
        'inverter_parameters': inverter,
    }
