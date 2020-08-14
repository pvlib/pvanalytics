"""Common fixtures for features tests."""
import pytest
import numpy as np
import pandas as pd

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
