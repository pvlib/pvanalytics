"""Tests for system paramter identification functions."""
import pytest
import pandas as pd
import numpy as np
from pvlib import irradiance
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
def solar_noon(solarposition_year):
    """Time series of solar noon in minutes on each day."""
    daytime = solarposition_year['apparent_zenith'] < 87
    minute = pd.Series(
        daytime.index.hour * 60 + daytime.index.minute,
        index=daytime.index
    )
    df = pd.DataFrame({'daytime': daytime, 'minute': minute})
    solar_noon = df.groupby(df.index.date).apply(
        lambda day: round(
            (day.minute[day.daytime].max() + day.minute[day.daytime].min()) / 2
        )
    )
    return solar_noon.reindex(pd.DatetimeIndex(solar_noon.index))


def test_longitude_solar_noon(solar_noon):
    longitude = system.longitude_solar_noon(solar_noon, utc_offset=-7)
    assert -110 < longitude < -100