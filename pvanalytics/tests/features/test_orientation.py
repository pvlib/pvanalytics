import pytest
from pandas.util.testing import assert_series_equal
import pandas as pd
from pvlib import pvsystem, tracking, modelchain, irradiance
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from pvanalytics.features import orientation


@pytest.fixture(scope='module')
def times():
    """Three days with one hour timestamp spacing in Etc/GMT+7"""
    return pd.date_range(
        start='03/01/2020',
        end='03/04/2020',
        closed='left',
        freq='H',
        tz='Etc/GMT+7'
    )


@pytest.fixture(scope='module')
def system_parameters():
    """System parameters for generating simulated power data."""
    sandia_modules = pvsystem.retrieve_sam('SandiaMod')
    sapm_inverters = pvsystem.retrieve_sam('cecinverter')
    module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
    inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
    temperature_model_parameters = (
        TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    )
    return {
        'module_parameters': module,
        'inverter_parameters': inverter,
        'temperature_model_parameters': temperature_model_parameters
    }


@pytest.fixture(scope='module')
def clearsky(times, albuquerque):
    """Clearsky at `times` in `albuquerque`."""
    return albuquerque.get_clearsky(times, model='simplified_solis')


@pytest.fixture(scope='module')
def solarposition(times, albuquerque):
    """Solar position at `times` in `albuquerque`."""
    return albuquerque.get_solarposition(times)


def test_clearsky_ghi_fixed(clearsky, solarposition):
    """Identify every day as fixed, since clearsky GHI is sunny."""
    assert orientation.fixed_nrel(
        clearsky['ghi'],
        solarposition['zenith'] < 87,
        r2_min=0.94,
    ).all()


@pytest.mark.filterwarnings("ignore:invalid value encountered in",
                            "ignore:divide by zero encountered in")
def test_perturbed_ghi_fixed(clearsky, solarposition):
    """If the clearsky for one day is perturbed then that day is not sunny."""
    ghi = clearsky['ghi']
    ghi.iloc[0:24] = 1
    expected = pd.Series(True, ghi.index)
    expected[0:24] = False
    assert_series_equal(
        expected,
        orientation.fixed_nrel(
            ghi,
            solarposition['zenith'] < 87
        ),
        check_names=False
    )


def test_ghi_not_tracking(clearsky, solarposition):
    """If we pass GHI measurements and tracking=True then no days are sunny."""
    assert (~orientation.tracking_nrel(
        clearsky['ghi'], solarposition['zenith'] < 87
    )).all()


@pytest.fixture
def power_tracking(clearsky, albuquerque, system_parameters):
    """Simulated power for a pvlib SingleAxisTracker PVSystem in Albuquerque"""
    system = tracking.SingleAxisTracker(**system_parameters)
    mc = modelchain.ModelChain(
        system,
        albuquerque,
        orientation_strategy='south_at_latitude_tilt'
    )
    mc.run_model(clearsky)
    return mc.ac


def test_power_tracking(power_tracking, solarposition):
    """simulated power from a single axis tracker is identified as sunny
    with tracking=True"""
    assert orientation.tracking_nrel(
        power_tracking,
        solarposition['zenith'] < 87
    ).all()


def test_power_tracking_perturbed(power_tracking, solarposition):
    """A day with perturbed values is not marked as tracking."""
    power_tracking.iloc[6:18] = 10
    expected = pd.Series(True, index=power_tracking.index)
    expected.iloc[0:24] = False
    assert_series_equal(
        expected,
        orientation.tracking_nrel(
            power_tracking,
            solarposition['zenith'] < 87
        )
    )
    assert_series_equal(
        expected,
        orientation.tracking_nrel(
            power_tracking,
            solarposition['zenith'] < 87,
            peak_min=100
        )
    )


def test_stuck_tracker_profile(solarposition, clearsky):
    """Test POA irradiance at a awkward orientation (high tilt and
    oriented West)."""
    poa = irradiance.get_total_irradiance(
        surface_tilt=45,
        surface_azimuth=270,
        **clearsky,
        solar_zenith=solarposition['apparent_zenith'],
        solar_azimuth=solarposition['azimuth']
    )
    assert not orientation.tracking_nrel(
        poa['poa_global'],
        solarposition['zenith'] < 87,
    ).any()
    # by restricting the data to the middle of the day (lower zenith
    # angles) we should classify the system as fixed
    assert orientation.fixed_nrel(
        poa['poa_global'],
        solarposition['zenith'] < 70
    ).all()
