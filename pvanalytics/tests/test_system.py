"""Tests for system paramter identification functions."""
import pandas as pd
from pvlib import irradiance
from pvanalytics import system


def _assert_within(x, expected, margin):
    # test that x is in the range [expected - margin, expected + margin]
    assert x in range(expected - margin, expected + margin + 1)


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
    _assert_within(azimuth, 180, 10)
    _assert_within(tilt, 15, 10)


def test_ghi_tilt_zero(clearsky_year, solarposition_year):
    """ghi has tilt equal to 0"""
    _, tilt = system.orientation(
        clearsky_year['ghi'],
        daytime=solarposition_year['apparent_zenith'] < 87,
        tilts=[0, 2, 4],
        azimuths=[180],
        solar_azimuth=solarposition_year['azimuth'],
        solar_zenith=solarposition_year['zenith'],
        **clearsky_year
    )
    assert tilt == 0


def test_azimuth_different_index(clearsky_year, solarposition_year,
                                 albuquerque):
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
        tilts=[40],
        azimuths=range(105, 135, 5),
        solar_azimuth=fine_solarposition['azimuth'],
        solar_zenith=fine_solarposition['apparent_zenith'],
        **fine_clearsky,
    )
    _assert_within(azimuth, 120, 10)
