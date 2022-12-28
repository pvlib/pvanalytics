"""Tests for system parameter identification functions."""
import pytest
import pandas as pd
import numpy as np
import pvlib
from pvlib import location, pvsystem, modelchain, irradiance
from pvanalytics import system


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
def summer_power_fixed(summer_clearsky, albuquerque, array_parameters,
                       system_parameters):
    """Simulated power from a FIXED PVSystem in Albuquerque, NM."""
    pv_system = pvsystem.PVSystem(surface_azimuth=180,
                                  surface_tilt=albuquerque.latitude,
                                  **array_parameters, **system_parameters)
    mc = modelchain.ModelChain(
        pv_system,
        albuquerque,
    )
    mc.run_model(summer_clearsky)
    ac = mc.results.ac
    return ac


@pytest.fixture
def summer_power_tracking(summer_clearsky, albuquerque, array_parameters,
                          system_parameters):
    """Simulated power for a TRACKING PVSystem in Albuquerque"""
    array = pvsystem.Array(pvsystem.SingleAxisTrackerMount(),
                           **array_parameters)
    system = pvsystem.PVSystem(arrays=[array],
                               **system_parameters)
    mc = modelchain.ModelChain(
        system,
        albuquerque
    )
    mc.run_model(summer_clearsky)
    return mc.results.ac


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
        clip_max=0.4
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


@pytest.fixture(scope='module')
def winter_perturbed(albuquerque_clearsky):
    winter_perturbed = albuquerque_clearsky.copy()
    winter = winter_perturbed.index.month.isin([1, 2, 3, 4, 10, 11, 12])
    winter_perturbed.loc[winter] = 10
    return winter_perturbed


@pytest.mark.filterwarnings("ignore:invalid value encountered in",
                            "ignore:divide by zero encountered in")
def test_year_bad_winter_tracking_envelope(winter_perturbed,
                                           albuquerque_clearsky):
    """If the data is perturbed during the winter months
    is_tracking_envelope() returns Tracker.UNKNOWN."""
    assert system.is_tracking_envelope(
        winter_perturbed['ghi'],
        albuquerque_clearsky['ghi'] > 0,
        pd.Series(False, index=albuquerque_clearsky.index)
    ) is system.Tracker.UNKNOWN


def test_tracking_envelope_seasonal_split(winter_perturbed,
                                          albuquerque_clearsky):
    """If winter or summer months are empty then only the season
    that is specified is used to determine fixed/tracking.

    Uses the fixture with perturbed winter months so that tests
    will fail if the winter months are analyzed.
    """
    # no 'summer' months
    assert system.is_tracking_envelope(
        winter_perturbed['ghi'],
        albuquerque_clearsky['ghi'] > 0,
        pd.Series(False, index=albuquerque_clearsky.index),
        seasonal_split={'summer': [], 'winter': [5, 6, 7, 8]}
    ) is system.Tracker.FIXED
    assert system.is_tracking_envelope(
        winter_perturbed['ghi'],
        albuquerque_clearsky['ghi'] > 0,
        pd.Series(False, index=albuquerque_clearsky.index),
        seasonal_split={'summer': None, 'winter': [5, 6, 7, 8]}
    ) is system.Tracker.FIXED
    # no 'winter' months
    assert system.is_tracking_envelope(
        winter_perturbed['ghi'],
        albuquerque_clearsky['ghi'] > 0,
        pd.Series(False, index=albuquerque_clearsky.index),
        seasonal_split={'summer': [5, 6, 7, 8], 'winter': []}
    ) is system.Tracker.FIXED
    assert system.is_tracking_envelope(
        winter_perturbed['ghi'],
        albuquerque_clearsky['ghi'] > 0,
        pd.Series(False, index=albuquerque_clearsky.index),
        seasonal_split={'summer': [5, 6, 7, 8], 'winter': None}
    ) is system.Tracker.FIXED
    # Leaving out a key is equivalent to passing None as its value
    assert system.is_tracking_envelope(
        winter_perturbed['ghi'],
        albuquerque_clearsky['ghi'] > 0,
        pd.Series(False, index=albuquerque_clearsky.index),
        seasonal_split={'summer': [5, 6, 7, 8]}
    ) is system.Tracker.FIXED
    assert system.is_tracking_envelope(
        winter_perturbed['ghi'],
        albuquerque_clearsky['ghi'] > 0,
        pd.Series(False, index=albuquerque_clearsky.index),
        seasonal_split={'winter': [5, 6, 7, 8]}
    ) is system.Tracker.FIXED
    # median fit should fail
    assert system.is_tracking_envelope(
        winter_perturbed['ghi'],
        albuquerque_clearsky['ghi'] > 0,
        pd.Series(False, index=albuquerque_clearsky.index),
        seasonal_split=None
    ) is system.Tracker.UNKNOWN
    assert system.is_tracking_envelope(
        winter_perturbed['ghi'],
        albuquerque_clearsky['ghi'] > 0,
        pd.Series(False, index=albuquerque_clearsky.index),
        seasonal_split=None,
        fit_median=False
    )
    # empty seasons should give Tracker.UNKNOWN
    with pytest.warns(UserWarning):
        assert system.is_tracking_envelope(
            albuquerque_clearsky['ghi'],
            albuquerque_clearsky['ghi'] > 0,
            pd.Series(False, index=albuquerque_clearsky.index),
            seasonal_split={'winter': [], 'summer': []}
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


@pytest.fixture(scope='module')
def naive_times():
    """One year at 1-hour intervals"""
    return pd.date_range(
        start='2020',
        end='2021',
        freq='H'
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


@pytest.fixture(scope='module',
                params=[(0, 180), (30, 180), (30, 90), (30, 270), (30, 0)],
                ids=['South-0', 'South-30', 'East-30', 'West-30', 'North-30'])
def system_power(request, system_location, naive_times):
    tilt = request.param[0]
    azimuth = request.param[1]
    local_time = naive_times.tz_localize(system_location.tz)
    clearsky = system_location.get_clearsky(
        local_time, model='simplified_solis'
    )
    solar_position = system_location.get_solarposition(local_time)
    poa = irradiance.get_total_irradiance(
        tilt, azimuth,
        solar_position['zenith'],
        solar_position['azimuth'],
        **clearsky
    )
    temp_cell = pvlib.temperature.sapm_cell(
        poa['poa_global'],
        25, 0,
        **pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[
            'sapm'
        ][
            'open_rack_glass_glass'
        ]
    )
    pdc = pvsystem.pvwatts_dc(poa['poa_global'], temp_cell, 100, -0.002)
    pac = pvsystem.inverter.pvwatts(pdc, 120)
    return {
        'location': system_location,
        'tilt': tilt,
        'azimuth': azimuth,
        'clearsky': clearsky,
        'solar_position': solar_position,
        'ac': pac
    }


@pytest.mark.slow
def test_orientation_fit_pvwatts(system_power):
    day_mask = system_power['ac'] > 0
    tilt, azimuth, rsquared = system.infer_orientation_fit_pvwatts(
        system_power['ac'][day_mask],
        solar_zenith=system_power['solar_position']['zenith'][day_mask],
        solar_azimuth=system_power['solar_position']['azimuth'][day_mask],
        **system_power['clearsky'][day_mask])
    assert rsquared > 0.9
    assert tilt == pytest.approx(system_power['tilt'], abs=10)
    if system_power['tilt'] == 0:
        # Any azimuth will give the same results at tilt 0.
        return
    if system_power['azimuth'] == 0:
        # 0 degrees equals 360 degrees.
        assert (azimuth == pytest.approx(0, abs=10)
                or azimuth == pytest.approx(360, abs=10))
    else:
        assert azimuth == pytest.approx(system_power['azimuth'], abs=10)


def test_orientation_fit_pvwatts_missing_data(naive_times):
    tilt = 30
    azimuth = 100
    system_location = location.Location(35, -106)
    local_time = naive_times.tz_localize('MST')
    clearsky = system_location.get_clearsky(
        local_time, model='simplified_solis'
    )
    clearsky.loc['3/1/2020':'3/15/2020'] = np.nan
    solar_position = system_location.get_solarposition(clearsky.index)
    solar_position.loc['3/1/2020':'3/15/2020'] = np.nan
    poa = irradiance.get_total_irradiance(
        tilt, azimuth,
        solar_position['zenith'],
        solar_position['azimuth'],
        **clearsky
    )
    temp_cell = pvlib.temperature.sapm_cell(
        poa['poa_global'],
        25, 0,
        **pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[
            'sapm'
        ][
            'open_rack_glass_glass'
        ]
    )
    pdc = pvsystem.pvwatts_dc(poa['poa_global'], temp_cell, 100, -0.002)
    pac = pvsystem.inverter.pvwatts(pdc, 120)
    solar_position.dropna(inplace=True)
    with pytest.raises(ValueError,
                       match=".* must not contain undefined values"):
        system.infer_orientation_fit_pvwatts(
            pac,
            solar_zenith=solar_position['zenith'],
            solar_azimuth=solar_position['azimuth'],
            **clearsky
        )
    pac.dropna(inplace=True)
    with pytest.raises(ValueError,
                       match=".* must not contain undefined values"):
        system.infer_orientation_fit_pvwatts(
            pac,
            solar_zenith=solar_position['zenith'],
            solar_azimuth=solar_position['azimuth'],
            **clearsky
        )
    clearsky.dropna(inplace=True)
    tilt_out, azimuth_out, rsquared = system.infer_orientation_fit_pvwatts(
        pac,
        solar_zenith=solar_position['zenith'],
        solar_azimuth=solar_position['azimuth'],
        **clearsky
    )
    assert rsquared > 0.9
    assert tilt_out == pytest.approx(tilt, abs=10)
    assert azimuth_out == pytest.approx(azimuth, abs=10)


def test_orientation_fit_pvwatts_temp_wind_as_series(naive_times):
    tilt = 30
    azimuth = 100
    system_location = location.Location(35, -106)
    local_time = naive_times.tz_localize('MST')
    clearsky = system_location.get_clearsky(
        local_time, model='simplified_solis'
    )
    solar_position = system_location.get_solarposition(clearsky.index)
    poa = irradiance.get_total_irradiance(
        tilt, azimuth,
        solar_position['zenith'],
        solar_position['azimuth'],
        **clearsky
    )
    temp_cell = pvlib.temperature.sapm_cell(
        poa['poa_global'],
        25, 1,
        **pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[
            'sapm'
        ][
            'open_rack_glass_glass'
        ]
    )
    temperature = pd.Series(25, index=clearsky.index)
    wind_speed = pd.Series(1, index=clearsky.index)
    temperature_missing = temperature.copy()
    temperature_missing.loc['4/5/2020':'4/10/2020'] = np.nan
    wind_speed_missing = wind_speed.copy()
    wind_speed_missing.loc['5/5/2020':'5/15/2020'] = np.nan
    pdc = pvsystem.pvwatts_dc(poa['poa_global'], temp_cell, 100, -0.002)
    pac = pvsystem.inverter.pvwatts(pdc, 120)
    with pytest.raises(ValueError,
                       match=".* must not contain undefined values"):
        system.infer_orientation_fit_pvwatts(
            pac,
            solar_zenith=solar_position['zenith'],
            solar_azimuth=solar_position['azimuth'],
            temperature=temperature_missing,
            wind_speed=wind_speed_missing,
            **clearsky
        )
    with pytest.raises(ValueError,
                       match="temperature must not contain undefined values"):
        system.infer_orientation_fit_pvwatts(
            pac,
            solar_zenith=solar_position['zenith'],
            solar_azimuth=solar_position['azimuth'],
            temperature=temperature_missing,
            wind_speed=wind_speed,
            **clearsky
        )
    with pytest.raises(ValueError,
                       match="wind_speed must not contain undefined values"):
        system.infer_orientation_fit_pvwatts(
            pac,
            solar_zenith=solar_position['zenith'],
            solar_azimuth=solar_position['azimuth'],
            temperature=temperature,
            wind_speed=wind_speed_missing,
            **clearsky
        )
    # ValueError if indices don't match
    with pytest.raises(ValueError):
        system.infer_orientation_fit_pvwatts(
            pac,
            solar_zenith=solar_position['zenith'],
            solar_azimuth=solar_position['azimuth'],
            temperature=temperature_missing.dropna(),
            wind_speed=wind_speed_missing.dropna(),
            **clearsky
        )
    tilt_out, azimuth_out, rsquared = system.infer_orientation_fit_pvwatts(
        pac,
        solar_zenith=solar_position['zenith'],
        solar_azimuth=solar_position['azimuth'],
        **clearsky
    )
    assert rsquared > 0.9
    assert tilt_out == pytest.approx(tilt, abs=10)
    assert azimuth_out == pytest.approx(azimuth, abs=10)
