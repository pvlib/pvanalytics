import pandas as pd
from pandas.testing import assert_series_equal
import pytest
from pvanalytics.features import shading


# TODO testing plan
#      - [ ] index of output same as input
#      - [ ] no shadows -> all False
#      - [ ] 5-minute timestamps
#      - [ ] unaligned timestamps (?)
#      - [ ] partial days at start and end of series


@pytest.fixture(scope='module')
def times():
    return pd.date_range(
        start='1/1/2020',
        end='12/31/2020',
        closed='left',
        freq='T',
        tz='MST'
    )


@pytest.fixture(scope='module')
def daytime(times, albuquerque):
    solar_position = albuquerque.get_solarposition(times)
    return solar_position['zenith'] < 87


@pytest.fixture(scope='module')
def clearsky_ghi(times, albuquerque):
    clearsky = albuquerque.get_clearsky(times, model='simplified_solis')
    return clearsky['ghi']


def test_fixed_no_shadows(daytime, clearsky_ghi):
    shadows, image = shading.fixed(clearsky_ghi, daytime, clearsky_ghi)
    assert not shadows.any()


def test_fixed_same_index(daytime, clearsky_ghi):
    daytime = daytime['1/2/2020 11:00':'11/2/2020 13:00']
    clearsky_ghi = clearsky_ghi['1/2/2020 11:00':'11/2/2020 13:00']
    shadows, image = shading.fixed(daytime, clearsky_ghi, clearsky_ghi)
    assert_series_equal(
        pd.series(False, index=daytime.index),
        shadows,
        check_names=False
    )
