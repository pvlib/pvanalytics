import pandas as pd
import numpy as np

import pytest
from pandas.util.testing import assert_series_equal

from pvanalytics.quality import irradiance


@pytest.fixture
def irradiance_qcrad():
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
    expected = irradiance_qcrad
    ghi_out_expected = expected['ghi_limit_flag']
    ghi_out = irradiance.check_ghi_limits_qcrad(expected['ghi'],
                                                expected['solar_zenith'],
                                                expected['dni_extra'])
    assert_series_equal(ghi_out, ghi_out_expected)


def test_check_dhi_limits_qcrad(irradiance_qcrad):
    expected = irradiance_qcrad
    dhi_out_expected = expected['dhi_limit_flag']
    dhi_out = irradiance.check_dhi_limits_qcrad(expected['dhi'],
                                                expected['solar_zenith'],
                                                expected['dni_extra'])
    assert_series_equal(dhi_out, dhi_out_expected)


def test_check_dni_limits_qcrad(irradiance_qcrad):
    expected = irradiance_qcrad
    dni_out_expected = expected['dni_limit_flag']
    dni_out = irradiance.check_dni_limits_qcrad(expected['dni'],
                                                expected['solar_zenith'],
                                                expected['dni_extra'])
    assert_series_equal(dni_out, dni_out_expected)


def test_check_irradiance_limits_qcrad(irradiance_qcrad):
    expected = irradiance_qcrad
    ghi_out_expected = expected['ghi_limit_flag']
    ghi_out, dhi_out, dni_out = irradiance.check_irradiance_limits_qcrad(
        expected['solar_zenith'], expected['dni_extra'], ghi=expected['ghi'])
    assert_series_equal(ghi_out, ghi_out_expected)
    assert dhi_out is None
    assert dni_out is None

    dhi_out_expected = expected['dhi_limit_flag']
    ghi_out, dhi_out, dni_out = irradiance.check_irradiance_limits_qcrad(
        expected['solar_zenith'], expected['dni_extra'], ghi=expected['ghi'],
        dhi=expected['dhi'])
    assert_series_equal(dhi_out, dhi_out_expected)

    dni_out_expected = expected['dni_limit_flag']
    ghi_out, dhi_out, dni_out = irradiance.check_irradiance_limits_qcrad(
        expected['solar_zenith'], expected['dni_extra'],
        dni=expected['dni'])
    assert_series_equal(dni_out, dni_out_expected)


def test_check_irradiance_consistency_qcrad(irradiance_qcrad):
    expected = irradiance_qcrad
    cons_comp, diffuse = irradiance.check_irradiance_consistency_qcrad(
        expected['ghi'], expected['solar_zenith'], expected['dni_extra'],
        expected['dhi'], expected['dni'])
    assert_series_equal(cons_comp, expected['consistent_components'])
    assert_series_equal(diffuse, expected['diffuse_ratio_limit'])


def test_check_limits():
    """Test the private check limits function."""
    expected = pd.Series(data=[True, False])
    data = pd.Series(data=[3, 2])
    result = irradiance._check_limits(val=data, lb=2.5)
    assert_series_equal(expected, result)
    result = irradiance._check_limits(val=data, lb=3, lb_ge=True)
    assert_series_equal(expected, result)

    data = pd.Series(data=[3, 4])
    result = irradiance._check_limits(val=data, ub=3.5)
    assert_series_equal(expected, result)
    result = irradiance._check_limits(val=data, ub=3, ub_le=True)
    assert_series_equal(expected, result)

    result = irradiance._check_limits(val=data, lb=3, ub=4, lb_ge=True,
                                      ub_le=True)
    assert all(result)
    result = irradiance._check_limits(val=data, lb=3, ub=4)
    assert not any(result)

    with pytest.raises(ValueError):
        irradiance._check_limits(val=data)
