"""Tests for funcitons that identify system characteristics."""
import pytest
import numpy as np
import pandas as pd
from pvlib import pvsystem, tracking, modelchain, irradiance
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from pvanalytics import system


@pytest.fixture
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
def summer_times():
    """One hour time stamps from May 1 through September 30, 2020 in GMT+7"""
    return pd.date_range(
        start='2020-5-1',
        end='2020-10-1',
        freq='H',
        closed='left',
        tz='Etc/GMT+7'
    )


@pytest.fixture(scope='module')
def summer_clearsky(summer_times, albuquerque):
    """Clearsky irradiance for `sumer_times` in Albuquerque, NM."""
    return albuquerque.get_clearsky(summer_times, model='simplified_solis')


@pytest.fixture
def summer_ghi(summer_clearsky):
    """Clearsky GHI for Summer, 2020 in Albuquerque, NM."""
    return summer_clearsky['ghi']


@pytest.fixture
def summer_power_fixed(summer_clearsky, albuquerque, system_parameters):
    """Simulated power from a FIXED PVSystem in Albuquerque, NM."""
    system = pvsystem.PVSystem(**system_parameters)
    mc = modelchain.ModelChain(
        system,
        albuquerque,
        orientation_strategy='south_at_latitude_tilt'
    )
    mc.run_model(summer_clearsky)
    return mc.ac


@pytest.fixture
def summer_power_tracking(summer_clearsky, albuquerque, system_parameters):
    """Simulated power for a pvlib SingleAxisTracker PVSystem in Albuquerque"""
    system = tracking.SingleAxisTracker(**system_parameters)
    mc = modelchain.ModelChain(
        system,
        albuquerque,
        orientation_strategy='south_at_latitude_tilt'
    )
    mc.run_model(summer_clearsky)
    return mc.ac


def test_ghi_tracking_envelope_fixed(summer_ghi):
    """Clearsky GHI for a system that is FIXED."""
    assert system.is_tracking_envelope(
        summer_ghi,
        summer_ghi > 0,
        pd.Series(False, index=summer_ghi.index)
    ) is system.Tracker.FIXED


def test_power_tracking_envelope_fixed(summer_power_fixed):
    """Simulated system under clearsky condidtions is FIXED."""
    assert system.is_tracking_envelope(
        summer_power_fixed,
        summer_power_fixed > 0,
        pd.Series(False, index=summer_power_fixed.index)
    ) is system.Tracker.FIXED


def test_power_tracking_envelope_tracking(summer_power_tracking):
    """Simulated single axis tracker is identified as TRACKING."""
    assert system.is_tracking_envelope(
        summer_power_tracking,
        summer_power_tracking > 0,
        pd.Series(False, index=summer_power_tracking.index)
    ) is system.Tracker.TRACKING


def test_high_clipping_unknown_tracking_envelope(summer_power_fixed):
    """If the amount of clipping is high then tracking is UNKNOWN"""
    clipping = pd.Series(False, index=summer_power_fixed.index)
    # 50% clipping
    clipping.iloc[0:len(clipping) // 2] = True
    assert system.is_tracking_envelope(
        summer_power_fixed,
        summer_power_fixed > 0,
        clipping,
        clip_max=40.0
    ) is system.Tracker.UNKNOWN


@pytest.mark.filterwarnings("ignore:invalid value encountered in",
                            "ignore:divide by zero encountered in")
def test_constant_unknown_tracking_envelope(summer_ghi):
    """A constant signal has unknown tracking."""
    constant = pd.Series(1, index=summer_ghi.index)
    assert system.is_tracking_envelope(
        constant,
        pd.Series(True, index=summer_ghi.index),
        pd.Series(False, index=summer_ghi.index),
    ) is system.Tracker.UNKNOWN


@pytest.mark.filterwarnings("ignore:invalid value encountered in",
                            "ignore:divide by zero encountered in")
def test_median_mismatch_tracking(summer_power_tracking):
    """If the median does not have the same fit as the 99.5% quantile then
    tracking is UNKNOWN."""
    power_half_tracking = summer_power_tracking.copy()
    power_half_tracking.iloc[0:100*24] = 1
    assert system.is_tracking_envelope(
        power_half_tracking,
        pd.Series(True, index=power_half_tracking.index),
        pd.Series(False, index=power_half_tracking.index),
        fit_median=False
    ) is system.Tracker.TRACKING
    assert system.is_tracking_envelope(
        power_half_tracking,
        pd.Series(True, index=power_half_tracking.index),
        pd.Series(False, index=power_half_tracking.index)
    ) is system.Tracker.UNKNOWN


@pytest.mark.filterwarnings("ignore:invalid value encountered in",
                            "ignore:divide by zero encountered in")
def test_median_mismatch_fixed(summer_power_fixed):
    """If the median does not have the same profile as the 99.5% quantile
    then tracking is UNKNOWN."""
    power_half_fixed = summer_power_fixed.copy()
    power_half_fixed.iloc[0:100*24] = 1
    assert system.is_tracking_envelope(
        power_half_fixed,
        pd.Series(True, index=power_half_fixed.index),
        pd.Series(False, index=power_half_fixed.index),
        fit_median=False
    ) is system.Tracker.FIXED
    assert system.is_tracking_envelope(
        power_half_fixed,
        pd.Series(True, index=power_half_fixed.index),
        pd.Series(False, index=power_half_fixed.index)
    ) is system.Tracker.UNKNOWN


def test_custom_tracking_envelope_thresholds(summer_power_fixed):
    """Can pass a custom set of minimal r^2 values."""
    assert system.is_tracking_envelope(
        summer_power_fixed,
        summer_power_fixed > 0,
        pd.Series(False, index=summer_power_fixed.index),
        fit_params={
            (0.5, 1.0): {'fixed': 0.9, 'tracking': 0.9, 'fixed_max': 0.9}
        }
    ) is system.Tracker.FIXED

    assert system.is_tracking_envelope(
        summer_power_fixed,
        summer_power_fixed > 0,
        pd.Series(False, index=summer_power_fixed.index),
        fit_params={
            (0.0, 1.0): {'fixed': 1.0, 'tracking': 0.8, 'fixed_max': 1.0}
        },
        fit_median=False
    ) is system.Tracker.TRACKING


@pytest.fixture(scope='module')
def albuquerque_clearsky(albuquerque):
    """One year of clearsky data in Albuquerque, NM."""
    year_hourly = pd.date_range(
        start='1/1/2020', end='1/1/2021', freq='H', tz='MST'
    )
    return albuquerque.get_clearsky(
        year_hourly,
        model='simplified_solis'
    )


def test_full_year_tracking_envelope(albuquerque_clearsky):
    """A full year of GHI should be identified as FIXED."""
    assert system.is_tracking_envelope(
        albuquerque_clearsky['ghi'],
        albuquerque_clearsky['ghi'] > 0,
        pd.Series(False, index=albuquerque_clearsky.index)
    ) is system.Tracker.FIXED


@pytest.mark.filterwarnings("ignore:invalid value encountered in",
                            "ignore:divide by zero encountered in")
def test_year_bad_winter_tracking_envelope(albuquerque_clearsky):
    """If the data is perturbed during the winter months
    is_tracking_envelope() returns Tracker.UNKNOWN."""
    winter_perturbed = albuquerque_clearsky.copy()
    winter = winter_perturbed.index.month.isin([10, 11, 12, 1, 2])
    winter_perturbed.loc[winter] = 10
    assert system.is_tracking_envelope(
        winter_perturbed['ghi'],
        albuquerque_clearsky['ghi'] > 0,
        pd.Series(False, index=albuquerque_clearsky.index)
    ) is system.Tracker.UNKNOWN


@pytest.fixture(scope='module')
def fine_index(clearsky_year):
    """DatetimeIndex for the same period as `clearsky_year` but with 1
    minute frequency."""
    return pd.date_range(
        start=clearsky_year.index.min(),
        end=clearsky_year.index.max(),
        freq='T'
    )


@pytest.fixture(scope='module')
def fine_clearsky(albuquerque, fine_index):
    """Clearsky in Albuquerque with 1 minute timestamp spacing."""
    return albuquerque.get_clearsky(
        fine_index,
        model='simplified_solis'
    )


@pytest.fixture(scope='module')
def fine_solarposition(albuquerque, fine_index):
    """Solar position in Albuquerque with 1 minute timestamp spacing."""
    return albuquerque.get_solarposition(fine_index)


def test_simple_poa_orientation(clearsky_year, solarposition_year,
                                fine_clearsky, fine_solarposition):
    poa = irradiance.get_total_irradiance(
        surface_tilt=15,
        surface_azimuth=180,
        **clearsky_year,
        solar_zenith=solarposition_year['apparent_zenith'],
        solar_azimuth=solarposition_year['azimuth']
    )
    azimuth, tilt = system.infer_orientation_daily_peak(
        poa['poa_global'],
        tilts=[15, 30],
        azimuths=[110, 180, 220],
        solar_zenith=fine_solarposition['apparent_zenith'],
        solar_azimuth=fine_solarposition['azimuth'],
        sunny=solarposition_year['apparent_zenith'] < 87,
        **fine_clearsky
    )
    assert azimuth == 180
    assert tilt == 15


def test_ghi_tilt_zero(clearsky_year, solarposition_year):
    """ghi has tilt equal to 0"""
    _, tilt = system.infer_orientation_daily_peak(
        clearsky_year['ghi'],
        sunny=solarposition_year['apparent_zenith'] < 87,
        tilts=[0, 5],
        azimuths=[180],
        solar_azimuth=solarposition_year['azimuth'],
        solar_zenith=solarposition_year['zenith'],
        **clearsky_year
    )
    assert tilt == 0


def test_azimuth_different_index(clearsky_year, solarposition_year,
                                 fine_clearsky, fine_solarposition):
    """Can use solar position and clearsky with finer time-resolution to
    get an accurate estimate of tilt and azimuth."""
    poa = irradiance.get_total_irradiance(
        surface_tilt=40,
        surface_azimuth=120,
        **clearsky_year,
        solar_zenith=solarposition_year['apparent_zenith'],
        solar_azimuth=solarposition_year['azimuth']
    )
    azimuth, tilt = system.infer_orientation_daily_peak(
        poa['poa_global'],
        sunny=solarposition_year['apparent_zenith'] < 87,
        tilts=[40],
        azimuths=[100, 120, 150],
        solar_azimuth=fine_solarposition['azimuth'],
        solar_zenith=fine_solarposition['apparent_zenith'],
        **fine_clearsky,
    )
    assert azimuth == 120


def test_orientation_with_gaps(clearsky_year, solarposition_year):
    poa = irradiance.get_total_irradiance(
        surface_tilt=15,
        surface_azimuth=180,
        **clearsky_year,
        solar_zenith=solarposition_year['apparent_zenith'],
        solar_azimuth=solarposition_year['azimuth']
    )
    poa.loc['2020-07-19':'2020-07-23'] = np.nan
    azimuth, tilt = system.infer_orientation_daily_peak(
        poa['poa_global'].dropna(),
        tilts=[15],
        azimuths=[180],
        solar_zenith=solarposition_year['apparent_zenith'],
        solar_azimuth=solarposition_year['azimuth'],
        sunny=solarposition_year['apparent_zenith'] < 87,
        **clearsky_year
    )
    assert azimuth == 180
    assert tilt == 15
