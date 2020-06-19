"""Tests for system paramter identification functions."""
import pandas as pd
from pvlib import irradiance
from pvanalytics import system


def test_simple_poa_orientation(clearsky_year, solarposition_year):
    poa = irradiance.get_total_irradiance(
        surface_tilt=15,
        surface_azimuth=180,
        **clearsky_year,
        solar_zenith=solarposition_year['apparent_zenith'],
        solar_azimuth=solarposition_year['azimuth']
    )
    azimuth, tilt = system.orientation(
        poa['poa_global'],
        tilts=range(5, 25, 5),
        azimuths=range(150, 200, 5),
        solar_zenith=solarposition_year['apparent_zenith'],
        solar_azimuth=solarposition_year['azimuth'],
        daytime=solarposition_year['apparent_zenith'] < 87,
        **clearsky_year
    )
    # May not get the exact azimuth, but it should be within 10
    # degrees of 180
    assert 170 < azimuth < 190
    # We should get the exact tilt since it is one of the candidates
    # we pass in to the function
    assert tilt == 15


def test_ghi_tilt_zero(clearsky_year, solarposition_year):
    """ghi has tilt equal to 0"""
    _, tilt = system.orientation(
        clearsky_year['ghi'],
        daytime=solarposition_year['apparent_zenith'] < 87,
        tilts=[0, 2, 4, 6, 8, 10],
        azimuths=[180],
        solar_azimuth=solarposition_year['azimuth'],
        solar_zenith=solarposition_year['zenith'],
        **clearsky_year
    )
    assert tilt == 0


def test_azimuth_different_index(clearsky_year, solarposition_year, albuquerque):
    """Can use solar position and clearsky with finer time-resolution to
    get an accurate estimate of tilt and azimuth."""
    poa = irradiance.get_total_irradiance(
        surface_tilt=40,
        surface_azimuth=120,
        **clearsky_year,
        solar_zenith=solarposition_year['apparent_zenith'],
        solar_azimuth=solarposition_year['azimuth']
    )
    fine_index = pd.date_range(
        start=clearsky_year.index.min(),
        end=clearsky_year.index.max(),
        freq='1T'
    )
    fine_solarposition = albuquerque.get_solarposition(fine_index)
    fine_clearsky = albuquerque.get_clearsky(
        fine_index,
        model='simplified_solis'
    )
    azimuth, tilt = system.orientation(
        poa['poa_global'],
        daytime=solarposition_year['apparent_zenith'] < 87,
        tilts=range(20, 50, 5),
        azimuths=range(100, 150, 5),
        solar_azimuth=fine_solarposition['azimuth'],
        solar_zenith=fine_solarposition['apparent_zenith'],
        **fine_clearsky,
    )
    assert azimuth == 120
    assert tilt == 40
