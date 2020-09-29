"""Tests for system parameter identification functions."""
import pytest
import pandas as pd
import numpy as np
from pvlib import irradiance, modelchain, pvsystem, location
from pvanalytics import system


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


@pytest.fixture(scope='module')
def solar_noon(albuquerque):
    """Time series of solar noon in minutes on each day."""
    days = pd.date_range(
        start='1/1/2020',
        end='12/31/2020',
        tz='MST',
        freq='D'
    )
    solar_noon = albuquerque.get_sun_rise_set_transit(
        days, method='spa'
    )['transit']
    return solar_noon


def test_longitude_solar_noon(solar_noon):
    """Can determine latitude within +/- 5 degrees"""
    longitude = system.longitude_solar_noon(
        solar_noon.dt.hour * 60 + solar_noon.dt.minute, utc_offset=-7
    )
    assert -111 < longitude < -101


@pytest.fixture(scope='module')
def clearsky_albuquerque(albuquerque):
    return albuquerque.get_clearsky(
        pd.date_range(
            start='1/1/2020',
            end='1/1/2021',
            freq='15T',
            closed='left',
            tz='MST'
        ),
        model='simplified_solis'
    )


@pytest.fixture(scope='module')
def power_albuquerque(albuquerque, clearsky_albuquerque, system_parameters):
    """One year of power output for a PV system in albuquerque NM.

    The system is oriented with azimuth=180 and tilt=30.
    """
    pv_system = pvsystem.PVSystem(tilt=30, azimuth=180, **system_parameters)
    mc = modelchain.ModelChain(
        pv_system,
        albuquerque,
    )
    mc.run_model(clearsky_albuquerque)
    return mc.ac


def test_orientation_haghdadi_value_error(power_albuquerque,
                                          clearsky_albuquerque):
    """Passing invalid parameter combinations raises a ValueError."""
    clearsky_conditions = power_albuquerque > 0
    # Passing neither longitude nor clearsky raises a ValueError
    with pytest.raises(ValueError,
                       match="longitude or clearsky_irradiance"
                             " must be specified"):
        system.infer_orientation_haghdadi(
            power_albuquerque,
            clearsky_conditions
        )
    # passing only latitude, but not longitude raises a ValueError
    with pytest.raises(ValueError,
                       match="longitude or clearsky_irradiance"
                             " must be specified"):
        system.infer_orientation_haghdadi(
            power_albuquerque,
            clearsky_conditions,
            latitude=35.5)
    # passing both longitude and clearsky raises a ValueError
    with pytest.raises(ValueError,
                       match="longitude and clearsky_irradiance"
                             " cannot both be specified"):
        system.infer_orientation_haghdadi(
            power_albuquerque,
            clearsky_conditions,
            clearsky_irradiance=clearsky_albuquerque,
            longitude=-106.5
        )


def test_infer_orientation_haghdadi_clearsky_irradiance(power_albuquerque,
                                                        clearsky_albuquerque):
    """Test that orientation can be inferred using user-supplied
    clearsky irradiance.

    In this case, the returned latitude should be None.
    """
    tilt, azimuth, latitude = system.infer_orientation_haghdadi(
        power_albuquerque,
        power_albuquerque > 0,
        clearsky_irradiance=clearsky_albuquerque
    )
    assert latitude is None
    # TODO verify that the tolerance below is reasonable
    assert tilt == pytest.approx(30, abs=5)
    assert azimuth == pytest.approx(180, abs=5)


def test_infer_orientation_haghdadi_fixed_latitude(albuquerque,
                                                   power_albuquerque):
    """If a specific latitude is passed as a parameter, that latitude
    is returned."""
    tilt, azimuth, latitude = system.infer_orientation_haghdadi(
        power_albuquerque,
        power_albuquerque > 0,
        longitude=albuquerque.longitude,
        latitude=albuquerque.latitude,
        clearsky_model='simplified_solis',
        tilt_estimate=45,
        azimuth_estimate=100
    )
    assert latitude == albuquerque.latitude
    # TODO verify that the tolerance below is reasonable
    assert tilt == pytest.approx(30, abs=5)
    assert azimuth == pytest.approx(180, abs=5)

@pytest.fixture(scope='module')
def naive_times():
    """One year at 1-hour intervals"""
    return pd.date_range(
        start='2020',
        end='2021',
        freq='H'
    )

@pytest.fixture(scope='module',
                params=[(0, 180), (30, 180), (30, 90), (30, 270), (30, 0)],
                ids=['South-0', 'South-30', 'East-30', 'West-30', 'North-30'])
def pv_system(request, system_parameters):
    return pvsystem.PVSystem(
        tilt=request.param[0],
        azimuth=request.param[1],
        **system_parameters
    )


@pytest.fixture(scope='module',
                params=[(35, -106, 'Etc/GMT+7'),
                        (50, 10, 'Etc/GMT-1'),
                        (-37, 144, 'Etc/GMT-10')],
                ids=['Albuquerque', 'Berlin', 'Melbourne'])
def system_location(request):
    """Location of the system."""
    return location.Location(
        request.param[0], request.param[1], tz=request.param[2]
    )


@pytest.fixture(scope='module')
def system_power(pv_system, system_location, naive_times):
    local_time = naive_times.tz_localize(system_location.tz)
    mc = modelchain.ModelChain(
        pv_system,
        system_location
    )
    clearsky = system_location.get_clearsky(
        local_time, model='simplified_solis'
    )
    mc.run_model(clearsky)
    return {
        'location': system_location,
        'tilt': pv_system.surface_tilt,
        'azimuth': pv_system.surface_azimuth,
        'ac': mc.ac
    }


def test_infer_orientation_haghdadi(system_power):
    tilt, azimuth, latitude = system.infer_orientation_haghdadi(
        system_power['ac'],
        system_power['ac'] > 0,
        longitude=system_power['location'].longitude,
        clearsky_model='simplified_solis'
    )
    assert tilt == pytest.approx(system_power['tilt'], abs=5)
    assert azimuth == pytest.approx(system_power['azimuth'], abs=5)
    assert latitude == pytest.approx(
        system_power['location'].latitude,
        abs=5
    )
