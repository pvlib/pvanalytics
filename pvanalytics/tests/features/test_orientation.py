import pytest
from pandas.util.testing import assert_series_equal
import pandas as pd
from pvlib import tracking, modelchain, irradiance
from pvanalytics.features import orientation


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


def test_frequency_to_hours():
    assert orientation._freqstr_to_hours('H') == 1.0
    assert orientation._freqstr_to_hours('15T') == 0.25
    assert orientation._freqstr_to_hours('2H') == 2.0
