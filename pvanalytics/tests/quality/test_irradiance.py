"""Tests for irradiance quality control functions."""
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
from pvlib.location import Location

import pytest
from pandas.util.testing import assert_series_equal

from pvanalytics.quality import irradiance


@pytest.fixture
def irradiance_qcrad():
    """Synthetic irradiance data and its expected quality flags."""
    output = pd.DataFrame(
        columns=['ghi', 'dhi', 'dni', 'solar_zenith', 'dni_extra',
                 'ghi_limit_flag', 'dhi_limit_flag', 'dni_limit_flag',
                 'consistent_components', 'diffuse_ratio_limit'],
        data=np.array([[-100, 100, 100, 30, 1370, 0, 1, 1, 0, 0],
                       [100, -100, 100, 30, 1370, 1, 0, 1, 0, 0],
                       [100, 100, -100, 30, 1370, 1, 1, 0, 0, 1],
                       [1000, 100, 900, 0, 1370, 1, 1, 1, 1, 1],
                       [1000, 200, 800, 15, 1370, 1, 1, 1, 1, 1],
                       [1000, 200, 800, 60, 1370, 0, 1, 1, 0, 1],
                       [1000, 300, 850, 80, 1370, 0, 0, 1, 0, 1],
                       [1000, 500, 800, 90, 1370, 0, 0, 1, 0, 1],
                       [500, 100, 1100, 0, 1370, 1, 1, 1, 0, 1],
                       [1000, 300, 1200, 0, 1370, 1, 1, 1, 0, 1],
                       [500, 600, 100, 60, 1370, 1, 1, 1, 0, 0],
                       [500, 600, 400, 80, 1370, 0, 0, 1, 0, 0],
                       [500, 500, 300, 80, 1370, 0, 0, 1, 1, 1],
                       [0, 0, 0, 93, 1370, 1, 1, 1, 0, 0]]))
    dtypes = ['float64', 'float64', 'float64', 'float64', 'float64',
              'bool', 'bool', 'bool', 'bool', 'bool']
    for (col, typ) in zip(output.columns, dtypes):
        output[col] = output[col].astype(typ)
    return output


def test_check_ghi_limits_qcrad(irradiance_qcrad):
    """Test that QCRad identifies out of bounds GHI values."""
    expected = irradiance_qcrad
    ghi_out_expected = expected['ghi_limit_flag']
    ghi_out = irradiance.check_ghi_limits_qcrad(expected['ghi'],
                                                expected['solar_zenith'],
                                                expected['dni_extra'])
    assert_series_equal(ghi_out, ghi_out_expected, check_names=False)


def test_check_dhi_limits_qcrad(irradiance_qcrad):
    """Test that QCRad identifies out of bounds DHI values."""
    expected = irradiance_qcrad
    dhi_out_expected = expected['dhi_limit_flag']
    dhi_out = irradiance.check_dhi_limits_qcrad(expected['dhi'],
                                                expected['solar_zenith'],
                                                expected['dni_extra'])
    assert_series_equal(dhi_out, dhi_out_expected, check_names=False)


def test_check_dni_limits_qcrad(irradiance_qcrad):
    """Test that QCRad identifies out of bounds DNI values."""
    expected = irradiance_qcrad
    dni_out_expected = expected['dni_limit_flag']
    dni_out = irradiance.check_dni_limits_qcrad(expected['dni'],
                                                expected['solar_zenith'],
                                                expected['dni_extra'])
    assert_series_equal(dni_out, dni_out_expected, check_names=False)


def test_check_irradiance_limits_qcrad(irradiance_qcrad):
    """Test different input combinations to check_irradiance_limits_qcrad."""
    expected = irradiance_qcrad
    ghi_out_expected = expected['ghi_limit_flag']
    ghi_out, dhi_out, dni_out = irradiance.check_irradiance_limits_qcrad(
        expected['solar_zenith'], expected['dni_extra'], ghi=expected['ghi'])
    assert_series_equal(ghi_out, ghi_out_expected, check_names=False)
    assert dhi_out is None
    assert dni_out is None

    dhi_out_expected = expected['dhi_limit_flag']
    ghi_out, dhi_out, dni_out = irradiance.check_irradiance_limits_qcrad(
        expected['solar_zenith'], expected['dni_extra'], ghi=expected['ghi'],
        dhi=expected['dhi'])
    assert_series_equal(dhi_out, dhi_out_expected, check_names=False)

    dni_out_expected = expected['dni_limit_flag']
    ghi_out, dhi_out, dni_out = irradiance.check_irradiance_limits_qcrad(
        expected['solar_zenith'], expected['dni_extra'],
        dni=expected['dni'])
    assert_series_equal(dni_out, dni_out_expected, check_names=False)


def test_check_irradiance_consistency_qcrad(irradiance_qcrad):
    """Test that QCRad identifies consistent irradiance measurements."""
    expected = irradiance_qcrad
    cons_comp, diffuse = irradiance.check_irradiance_consistency_qcrad(
        expected['ghi'], expected['solar_zenith'],
        expected['dhi'], expected['dni'])
    assert_series_equal(cons_comp, expected['consistent_components'],
                        check_names=False)
    assert_series_equal(diffuse, expected['diffuse_ratio_limit'],
                        check_names=False)


@pytest.fixture
def location():
    """Fixture giving Albuquerque's location."""
    return Location(
        latitude=35.05,
        longitude=-106.5,
        altitude=1619,
        name="Albuquerque",
        tz="MST"
    )


@pytest.fixture
def times():
    """One hour of times at 10 minute frequency."""
    mst = pytz.timezone('MST')
    return pd.date_range(
        start=datetime(2018, 6, 15, 12, 0, 0, tzinfo=mst),
        end=datetime(2018, 6, 15, 13, 0, 0, tzinfo=mst),
        freq='10T'
    )


def test_ghi_clearsky_limits(location, times):
    """GHI values greater than clearsky values are flagged False."""
    clearsky = location.get_clearsky(times)
    ghi = clearsky['ghi'].copy()
    ghi.iloc[0] *= 0.5
    ghi.iloc[-1] *= 2.0
    clear_times = np.tile(True, len(times))
    clear_times[-1] = False
    assert_series_equal(
        irradiance.ghi_clearsky_limits(ghi, clearsky['ghi']),
        pd.Series(index=times, data=clear_times)
    )


def test_poa_clearsky_limits():
    """POA irradiance values greater than clearsky valuse are flagged False."""
    index = pd.date_range(start=datetime(2019, 6, 15, 12, 0, 0),
                          freq='15T', periods=5)
    poa = pd.Series(index=index, data=[800, 1000, 1200, -200, np.nan])
    poa_clearsky = pd.Series(index=index, data=1000)
    assert_series_equal(
        irradiance.poa_clearsky_limits(poa, poa_clearsky),
        pd.Series(index=index, data=[True, True, False, True, False])
    )
    assert_series_equal(
        irradiance.poa_clearsky_limits(poa, poa_clearsky, csi_max=1.2),
        pd.Series(index=index, data=[True, True, True, True, False])
    )
