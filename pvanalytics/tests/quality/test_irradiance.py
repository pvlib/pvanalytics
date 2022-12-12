"""Tests for irradiance quality control functions."""
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
import pytest
from pandas.util.testing import assert_series_equal
from pvanalytics.quality import irradiance
from ..conftest import DATA_DIR


test_file_1 = DATA_DIR / "irradiance_RMIS_NREL.csv"


@pytest.fixture
def generate_RMIS_irradiance_series():
    # Pull down the saved PVLib dataframe and process it
    df = pd.read_csv(test_file_1, index_col=0, parse_dates=True)
    df = df.tz_localize("Etc/GMT+7")
    # Get the GHI, DHI, and DNI series
    dni_series = df['irradiance_dni__7982']
    dhi_series = df['irradiance_dhi__7983']
    ghi_series = df['irradiance_ghi__7981']
    dni_clear_series = df['pvlib_clearsky_dni']
    sza = df['pvlib_zenith']
    return (dhi_series, dni_series, ghi_series, dni_clear_series, sza)


@pytest.fixture
def irradiance_qcrad():
    """Synthetic irradiance data and its expected quality flags.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
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
    """Test that QCRad identifies out of bounds GHI values.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    expected = irradiance_qcrad
    ghi_out_expected = expected['ghi_limit_flag']
    ghi_out = irradiance.check_ghi_limits_qcrad(expected['ghi'],
                                                expected['solar_zenith'],
                                                expected['dni_extra'])
    assert_series_equal(ghi_out, ghi_out_expected, check_names=False)


def test_check_dhi_limits_qcrad(irradiance_qcrad):
    """Test that QCRad identifies out of bounds DHI values.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    expected = irradiance_qcrad
    dhi_out_expected = expected['dhi_limit_flag']
    dhi_out = irradiance.check_dhi_limits_qcrad(expected['dhi'],
                                                expected['solar_zenith'],
                                                expected['dni_extra'])
    assert_series_equal(dhi_out, dhi_out_expected, check_names=False)


def test_check_dni_limits_qcrad(irradiance_qcrad):
    """Test that QCRad identifies out of bounds DNI values.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    expected = irradiance_qcrad
    dni_out_expected = expected['dni_limit_flag']
    dni_out = irradiance.check_dni_limits_qcrad(expected['dni'],
                                                expected['solar_zenith'],
                                                expected['dni_extra'])
    assert_series_equal(dni_out, dni_out_expected, check_names=False)


def test_check_irradiance_limits_qcrad(irradiance_qcrad):
    """Test different input combinations to check_irradiance_limits_qcrad.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
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
    """Test that QCRad identifies consistent irradiance measurements.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    expected = irradiance_qcrad
    cons_comp, diffuse = irradiance.check_irradiance_consistency_qcrad(
        expected['solar_zenith'], expected['ghi'],
        expected['dhi'], expected['dni'])
    assert_series_equal(cons_comp, expected['consistent_components'],
                        check_names=False)
    assert_series_equal(diffuse, expected['diffuse_ratio_limit'],
                        check_names=False)


@pytest.fixture
def times():
    """One hour of times at 10 minute frequency.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    mst = pytz.timezone('MST')
    return pd.date_range(
        start=datetime(2018, 6, 15, 12, 0, 0, tzinfo=mst),
        end=datetime(2018, 6, 15, 13, 0, 0, tzinfo=mst),
        freq='10T'
    )


def test_clearsky_limits(times):
    """Values greater than clearsky values are flagged False.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    clearsky = pd.Series(np.linspace(50, 55, len(times)), index=times)
    measured = clearsky.copy()
    measured.iloc[0] *= 0.5
    measured.iloc[-1] *= 2.0
    clear_times = np.tile(True, len(times))
    clear_times[-1] = False
    assert_series_equal(
        irradiance.clearsky_limits(measured, clearsky),
        pd.Series(index=times, data=clear_times)
    )


def test_clearsky_limits_negative_and_nan():
    """Irradiance values greater than clearsky valuse are flagged False
    along with NaNs.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    index = pd.date_range(start=datetime(2019, 6, 15, 12, 0, 0),
                          freq='15T', periods=5)
    measured = pd.Series(index=index, data=[800, 1000, 1200, -200, np.nan])
    clearsky = pd.Series(index=index, data=1000)
    assert_series_equal(
        irradiance.clearsky_limits(measured, clearsky),
        pd.Series(index=index, data=[True, True, False, True, False])
    )


def test_clearsky_limits_csi_max(times):
    """Increasing `csi_max` passes larger values."""
    measured = pd.Series(np.linspace(800, 1000, len(times)), index=times)
    measured.iloc[0] = 1300
    measured.iloc[1] = 1200
    measured.iloc[2] = 1100
    clearsky = pd.Series(1000, index=times)
    expected = pd.Series(True, index=times)
    expected.iloc[0:3] = False
    assert_series_equal(
        irradiance.clearsky_limits(measured, clearsky, csi_max=1.0),
        expected
    )
    expected.iloc[2] = True
    assert_series_equal(
        irradiance.clearsky_limits(measured, clearsky, csi_max=1.1),
        expected
    )
    expected.iloc[1] = True
    assert_series_equal(
        irradiance.clearsky_limits(measured, clearsky, csi_max=1.2),
        expected
    )


def test_daily_insolation_limits(albuquerque):
    """Daily insolation limits works with uniform timestamp spacing."""
    three_days = pd.date_range(
        start='1/1/2020',
        end='1/4/2020',
        closed='left',
        freq='H'
    )
    clearsky = albuquerque.get_clearsky(three_days, model='simplified_solis')
    assert irradiance.daily_insolation_limits(
        clearsky['ghi'], clearsky['ghi']
    ).all()
    assert not irradiance.daily_insolation_limits(
        pd.Series(0, index=three_days),
        clearsky['ghi']
    ).any()
    irrad = clearsky['ghi'].copy()
    irrad.loc['1/1/2020'] = irrad['1/1/2020']*0.7
    irrad.loc['1/2/2020'] = irrad['1/2/2020']*2.0
    irrad.loc['1/3/2020'] = irrad['1/3/2020']*0.3
    expected = pd.Series(False, irrad.index)
    expected.loc['1/1/2020'] = True
    assert_series_equal(
        expected,
        irradiance.daily_insolation_limits(irrad, clearsky['ghi']),
        check_names=False
    )


def test_daily_insolation_limits_uneven(albuquerque):
    """daily_insolation_limits works with uneven timestamp spacing."""
    three_days = pd.date_range(
        start='1/1/2020',
        end='1/4/2020',
        closed='left',
        freq='15T'
    )
    clearsky = albuquerque.get_clearsky(three_days, model='simplified_solis')
    ghi = clearsky['ghi'].copy()
    assert_series_equal(
        irradiance.daily_insolation_limits(ghi, ghi),
        pd.Series(True, three_days),
        check_names=False
    )
    ghi.loc[(ghi > 200) & (ghi < 202)] = np.nan
    ghi.loc[ghi == 0] = np.nan
    ghi.dropna(inplace=True)
    assert_series_equal(
        irradiance.daily_insolation_limits(
            ghi,
            albuquerque.get_clearsky(
                ghi.index,
                model='simplified_solis'
            )['ghi']
        ),
        pd.Series(True, index=ghi.index),
        check_names=False
    )


def test_calculate_ghi_component(generate_RMIS_irradiance_series):
    """
    Test calculate_component_sum_series() function on GHI calculation.
    """
    # Pull down RMIS data to test on
    dhi_series, dni_series, ghi_series, dni_clear_series, sza_series = \
        generate_RMIS_irradiance_series
    # Run with fill_night_value as np.nan
    ghi_series_fill_value = irradiance.calculate_component_sum_series(
        solar_zenith=sza_series,
        dhi=dhi_series,
        dni=dni_series,
        zenith_limit=90,
        fill_night_value=np.nan)
    # Make sure that periods where sza>90 are marked as NaN
    assert all(ghi_series_fill_value[sza_series > 90].isna())
    # Run with fill_night_value = 'equation'
    ghi_series_equation = irradiance.calculate_component_sum_series(
        solar_zenith=sza_series,
        dhi=dhi_series,
        dni=dni_series,
        fill_night_value='equation')
    # Make sure that periods where sza>90 are equal equal to GHI values
    assert all(ghi_series_equation[sza_series > 90].dropna() ==
               dhi_series[sza_series > 90].dropna())
    # Run with fill_night_value = None
    ghi_series_none = irradiance.calculate_component_sum_series(
        solar_zenith=sza_series,
        dhi=dhi_series,
        dni=dni_series,
        zenith_limit=90,
        fill_night_value=None)
    ghi_test = dni_series * np.cos(sza_series * np.pi / 180) + dhi_series
    assert all(ghi_test.round(5).dropna() == ghi_series_none.round(5).dropna())
    # Throw an error if int/float, None, or 'equation' aren't passed in
    # fill_night_value parameter
    with pytest.raises(ValueError):
        irradiance.calculate_component_sum_series(solar_zenith=sza_series,
                                                  dhi=dhi_series,
                                                  dni=dni_series,
                                                  zenith_limit=90,
                                                  fill_night_value='random')


def test_calculate_dhi_component(generate_RMIS_irradiance_series):
    """
    Test calculate_component_sum_series() function on DHI calculation.
    """
    # Pull down RMIS data to test on
    dhi_series, dni_series, ghi_series, dni_clear_series, sza_series = \
        generate_RMIS_irradiance_series
    # Test with equation used
    dhi_series_equation = irradiance.calculate_component_sum_series(
        solar_zenith=sza_series,
        ghi=ghi_series,
        dni=dni_series,
        zenith_limit=90,
        fill_night_value='equation')
    # Make sure that periods where sza>90 are equal equal to GHI values
    assert all(dhi_series_equation[sza_series > 90].dropna() ==
               ghi_series[sza_series > 90].dropna())


def test_calculate_dni_component(generate_RMIS_irradiance_series):
    """
    Test calculate_component_sum_series() function on DNI calculation.
    """
    # Pull down RMIS data to test on
    dhi_series, dni_series, ghi_series, dni_clear_series, sza_series = \
        generate_RMIS_irradiance_series
    dni_series_equation = irradiance.calculate_component_sum_series(
        solar_zenith=sza_series,
        ghi=ghi_series,
        dhi=dhi_series,
        dni_clear=dni_clear_series,
        zenith_limit=90,
        fill_night_value='equation')
    # Make sure that periods where sza>90 are equal equal to GHI values
    assert all(dni_series_equation[sza_series > 90].dropna() == 0)


@pytest.fixture
def lorenz_test_data():

    data = pd.DataFrame(
        columns=['ghi', 'poa_global', 'solar_zenith', 'azimuth', 'aoi',
                 'lower_limit_poa', 'upper_limit_poa',
                 'poa_global_limit_int_flag', 'poa_global_limit_bool_flag',
                 'lower_limit_ghi', 'upper_limit_flag2_ghi',
                 'upper_limit_flag3_ghi', 'ghi_limit_int_flag',
                 'ghi_limit_bool_flag'],
        data=np.array([[400, 730, 60, 205, 30, 6.835, 1335.256062, 0, 1,
                        6.835, 870.2, 992.531965, 0, 1],
                       [500, 830, 60, 205, 30, 6.835, 1335.256062, 0, 1,
                        6.835, 870.2, 992.531965, 0, 1],
                       [880, 830, 60, 205, 30, 6.835, 1335.256062, 0, 1,
                        6.835, 870.2, 992.531965, 2, 0],
                       [1000, 830, 60, 205, 30, 6.835, 1335.256062, 0, 1,
                        6.835, 870.2, 992.531965, 3, 0],
                       [6, 830, 60, 205, 30, 6.835, 1335.256062, 0, 1,
                        6.835, 870.2, 992.531965, 3, 0],
                       [100, 150, 60, 205, 30, 6.835, 1335.256062, 0, 1,
                        6.835, 870.2, 992.531965, 0, 1],
                       [100, 150, 60, 205, 30, 6.835, 1335.256062, 0, 1,
                        6.835, 870.2, 992.531965, 0, 1],
                       [100, 1152, 60, 205, 30, 6.835, 1335.256062, 3, 0,
                        6.835, 870.2, 992.531965, 0, 1],
                       [100, 150, 60, 205, 30, 6.835, 1335.256062, 3, 0,
                        6.835, 870.2, 992.531965, 0, 1],
                       [100, 150, 60, 205, 30, 6.835, 1335.256062, 0, 1,
                        6.835, 870.2, 992.531965, 0, 1],
                       [500, 830, np.nan, 205, 30, np.nan, np.nan, 1, 0,
                        np.nan, np.nan, np.nan, 1, 0],
                       [500, 830, 60, 205, np.nan, 6.835, np.nan, 1, 0,
                        6.835, 870.2, 992.531965, 0, 1]]))

    dtypes = ['float64', 'float64', 'float64', 'float64', 'float64',
              'float64', 'float64',
              'int64', 'bool',
              'float64', 'float64',
              'float64', 'int64',
              'bool']

    for (col, typ) in zip(data.columns, dtypes):
        data[col] = data[col].astype(typ)

    return(data)


def test_check_poa_global_limits_lorenz(lorenz_test_data):
    """Testing the function 'check_poa_global_limits_lorenz'"""

    data = lorenz_test_data

    # Expected boolean and integer flags
    expected_bool_flag = data['poa_global_limit_bool_flag']
    expected_int_flag = data['poa_global_limit_int_flag']

    # Setting up inputs
    dni_extra = 1367
    poa_global = data['poa_global']
    solar_zenith = data['solar_zenith']
    aoi = data['aoi']

    # Calling the functions
    poa_global_limit_bool_flag, poa_global_limit_int_flag = \
        irradiance.check_poa_global_limits_lorenz(poa_global, solar_zenith,
                                                  aoi, dni_extra)

    assert_series_equal(expected_int_flag,
                        poa_global_limit_int_flag,
                        check_names=False)

    assert_series_equal(poa_global_limit_bool_flag,
                        expected_bool_flag,
                        check_names=False)


def test__upper_poa_global_limit_lorenz(lorenz_test_data):
    """Testing upper poa global limit defined by Lorenz et. al"""

    data = lorenz_test_data

    # Expected upper limit
    expected_upper_limit = data['upper_limit_poa']

    # Setting up inputs
    dni_extra = 1367
    solar_zenith = data['solar_zenith']
    aoi = data['aoi']

    # Testing upper limit
    test_upper_limit = irradiance._upper_poa_global_limit_lorenz(aoi,
                                                                 solar_zenith,
                                                                 dni_extra)

    assert_series_equal(expected_upper_limit,
                        test_upper_limit,
                        check_names=False)


def test__lower_poa_global_limit_lorenz(lorenz_test_data):
    """Testing lower poa global limit defined by Lorenz et. al"""

    data = lorenz_test_data

    # Expected upper limit
    expected_lower_limit = data['lower_limit_poa']

    # Setting up inputs
    dni_extra = 1367
    solar_zenith = data['solar_zenith']

    test_lower_limit = irradiance._lower_poa_global_limit_lorenz(solar_zenith,
                                                                 dni_extra)

    assert_series_equal(expected_lower_limit,
                        test_lower_limit,
                        check_names=False)


def test__upper_ghi_limit_lorenz_flag2(lorenz_test_data):
    """Testing upper ghi limit for flag 2 defined by Lorenz et. al"""

    data = lorenz_test_data

    # Expected upper limit
    expected_upper_limit = data['upper_limit_flag2_ghi']

    # Setting up inputs
    dni_extra = 1367
    solar_zenith = data['solar_zenith']

    # Testing upper limit
    test_upper_limit = irradiance._upper_ghi_limit_lorenz_flag2(solar_zenith,
                                                                dni_extra)

    assert_series_equal(expected_upper_limit,
                        test_upper_limit,
                        check_names=False)


def test__upper_ghi_limit_lorenz_flag3(lorenz_test_data):
    """Testing upper ghi limit for flag 3 defined by Lorenz et. al"""

    data = lorenz_test_data

    # Expected upper limit
    expected_upper_limit = data['upper_limit_flag3_ghi']

    # Setting up inputs
    dni_extra = 1367
    solar_zenith = data['solar_zenith']

    # Testing upper limit
    test_upper_limit = irradiance._upper_ghi_limit_lorenz_flag3(solar_zenith,
                                                                dni_extra)

    assert_series_equal(expected_upper_limit,
                        test_upper_limit,
                        check_names=False)


def test__lower_ghi_limit_lorenz(lorenz_test_data):
    """Testing lower ghi limit defined by Lorenz et. al"""

    data = lorenz_test_data

    # Expected upper limit
    expected_upper_limit = data['lower_limit_ghi']

    # Setting up inputs
    dni_extra = 1367
    solar_zenith = data['solar_zenith']

    # Testing upper limit
    test_upper_limit = irradiance._lower_ghi_limit_lorenz(solar_zenith,
                                                          dni_extra)

    assert_series_equal(expected_upper_limit,
                        test_upper_limit,
                        check_names=False)


def test_check_ghi_limits_lorenz(lorenz_test_data):
    """Testing the function 'check_ghi_limits_lorenz'"""

    data = lorenz_test_data

    # Expected boolean and integer flags
    expected_bool_flag = data['ghi_limit_bool_flag']
    expected_int_flag = data['ghi_limit_int_flag']

    # Setting up inputs
    dni_extra = 1367
    ghi = data['ghi']
    solar_zenith = data['solar_zenith']

    # Calling the functions
    ghi_limit_bool_flag, ghi_limit_int_flag = \
        irradiance.check_ghi_limits_lorenz(ghi, solar_zenith, dni_extra)

    assert_series_equal(expected_int_flag,
                        ghi_limit_int_flag,
                        check_names=False)

    assert_series_equal(ghi_limit_bool_flag,
                        expected_bool_flag,
                        check_names=False)
