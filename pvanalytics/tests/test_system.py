"""Tests for funcitons that identify system characteristics."""
import pytest
import pandas as pd
from pvlib import location, pvsystem, tracking, modelchain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from pvanalytics import system


# Rough testing plan
#
# Generate several data sets (winter/summer only) using PVLib with
# different orientation and other characteristics and validate
# system.orientation by making sure the correct orientation is
# inferred.

# TODO Clearsky POA should be identifified as FIXED

# TODO Generate data using pvlib.tracking.SingleAxisTracker (TRACKING)

# TODO Address data with wrong timezone (midnight is during th day)


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
def albuquerque():
    """pvlib Location for Albuquerque, NM."""
    return location.Location(
        35.0844,
        -106.6504,
        name='Albuquerque',
        altitude=1500,
        tx='Etc/GMT+7'
    )


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


def test_ghi_orientation_fixed(summer_ghi):
    """Clearsky GHI for has a FIXED Orientation."""
    assert system.orientation(
        summer_ghi,
        summer_ghi > 0,
        pd.Series(False, index=summer_ghi.index)
    ) is system.Orientation.FIXED


def test_power_orientation_fixed(summer_power_fixed):
    """Simulated system under clearsky condidtions is FIXED."""
    assert system.orientation(
        summer_power_fixed,
        summer_power_fixed > 0,
        pd.Series(False, index=summer_power_fixed.index)
    ) is system.Orientation.FIXED


def test_power_orientation_tracking(summer_power_tracking):
    """Simulated single axis tracker is identifified as TRACKING."""
    assert system.orientation(
        summer_power_tracking,
        summer_power_tracking > 0,
        pd.Series(False, index=summer_power_tracking.index)
    ) is system.Orientation.TRACKING


def test_high_clipping_unknown_orientation(summer_power_fixed):
    """If the amount of clipping is high then orientation is UNKNOWN"""
    clipping = pd.Series(False, index=summer_power_fixed.index)
    # 50% clipping
    clipping.iloc[0:len(clipping) // 2] = True
    assert system.orientation(
        summer_power_fixed,
        summer_power_fixed > 0,
        clipping,
        clip_max=40.0
    ) is system.Orientation.UNKNOWN


def test_constant_unknown_orientation(summer_ghi):
    """A constant signal has unknown orientation."""
    constant = pd.Series(True, index=summer_ghi.index)
    assert system.orientation(
        constant,
        pd.Series(True, index=summer_ghi.index),
        pd.Series(False, index=summer_ghi.index),
    ) is system.Orientation.UNKNOWN
