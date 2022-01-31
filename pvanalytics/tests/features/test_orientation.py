import pytest
from pandas.util.testing import assert_series_equal
import pandas as pd
from pvlib import tracking, pvsystem, modelchain, irradiance
from pvanalytics.features import orientation

from ..conftest import requires_pvlib


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
def power_tracking_old_pvlib(clearsky, albuquerque, array_parameters,
                             system_parameters):
    """Simulated power for a pvlib SingleAxisTracker PVSystem in Albuquerque"""
    # copy of `power_tracking` but with older pvlib API
    # TODO: remove when minimum pvlib version is >= 0.9.0
    system = tracking.SingleAxisTracker(**array_parameters,
                                        **system_parameters)
    mc = modelchain.ModelChain(
        system,
        albuquerque,
    )
    mc.run_model(clearsky)
    return mc.ac


@pytest.fixture
def power_tracking(clearsky, albuquerque, array_parameters, system_parameters):
    """Simulated power for a pvlib SingleAxisTracker PVSystem in Albuquerque"""
    array = pvsystem.Array(pvsystem.SingleAxisTrackerMount(),
                           **array_parameters)
    system = pvsystem.PVSystem(arrays=[array],
                               **system_parameters)
    mc = modelchain.ModelChain(
        system,
        albuquerque,
    )
    mc.run_model(clearsky)
    return mc.results.ac


@requires_pvlib('<0.9.0', reason="SingleAxisTracker deprecation")
def test_power_tracking_old_pvlib(power_tracking_old_pvlib, solarposition):
    """simulated power from a single axis tracker is identified as sunny
    with tracking=True"""
    # copy of `test_power_tracking` but with older pvlib API
    # TODO: remove when minimum pvlib version is >= 0.9.0
    assert orientation.tracking_nrel(
        power_tracking_old_pvlib,
        solarposition['zenith'] < 87
    ).all()


@requires_pvlib('>=0.9.0', reason="Array class")
def test_power_tracking(power_tracking, solarposition):
    """simulated power from a single axis tracker is identified as sunny
    with tracking=True"""
    assert orientation.tracking_nrel(
        power_tracking,
        solarposition['zenith'] < 87
    ).all()


@requires_pvlib('<0.9.0', reason="SingleAxisTracker deprecation")
def test_power_tracking_perturbed_old_pvlib(power_tracking_old_pvlib,
                                            solarposition):
    """A day with perturbed values is not marked as tracking."""
    # copy of `test_power_tracking_perturbed` but with older pvlib API
    # TODO: remove when minimum pvlib version is >= 0.9.0
    power_tracking_old_pvlib.iloc[6:18] = 10
    expected = pd.Series(True, index=power_tracking_old_pvlib.index)
    expected.iloc[0:24] = False
    assert_series_equal(
        expected,
        orientation.tracking_nrel(
            power_tracking_old_pvlib,
            solarposition['zenith'] < 87
        )
    )
    assert_series_equal(
        expected,
        orientation.tracking_nrel(
            power_tracking_old_pvlib,
            solarposition['zenith'] < 87,
            peak_min=100
        )
    )


@requires_pvlib('>=0.9.0', reason="Array class")
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
