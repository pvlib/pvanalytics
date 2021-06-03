import pandas as pd
from pandas.testing import assert_series_equal
import pytest
from pvanalytics.features import shading


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
    shadows, image = shading.fixed(clearsky_ghi, daytime, clearsky_ghi)
    assert_series_equal(
        pd.Series(False, index=daytime.index),
        shadows,
        check_names=False
    )


def test_simple_shadow(daytime, clearsky_ghi):
    shadow_ghi = clearsky_ghi.copy()
    shadow_ghi[shadow_ghi.between_time('11:00', '11:03').index] *= 0.5
    shadows, image = shading.fixed(shadow_ghi, daytime, clearsky_ghi)
    assert shadows.between_time('11:00', '11:03').all()


def test_invalid_interval(daytime, clearsky_ghi):
    ghi = clearsky_ghi.resample('5T').first()
    daytime_resampled = daytime.resample('5T').first()
    with pytest.raises(ValueError, match="Data must be at 1-minute intervals"):
        shading.fixed(ghi, daytime_resampled, ghi)
    with pytest.raises(ValueError, match="Data must be at 1-minute intervals"):
        shading.fixed(daytime, clearsky_ghi, clearsky_ghi, interval=2)
