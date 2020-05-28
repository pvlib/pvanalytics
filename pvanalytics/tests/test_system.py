"""Tests for funcitons that identify system characteristics."""
import pytest
import pandas as pd
from pvlib import location
from pvlib import irradiance
from pvanalytics import system


# TODO testing plan for system.orientation
#
# Generate several data sets (winter/summer only) using PVLib with
# different orientation and other characteristics and validate
# system.orientation by making sure the correct orientation is
# inferred.
#
# - Do this for GHI and POA which should both be identifified as
#   Orientation.FIXED
#
# - Generate data using pvlib.tracking.SingleAxisTracker
#
# - Generate power data from a PVSystem without a a tracker


@pytest.fixture
def summer_times():
    """ten-minute time stamps from 1 May through 31 September in GMT+7 time"""
    return pd.date_range(
        start='2020-5-1',
        end='2020-10-1',
        freq='10T',
        closed='left',
        tz='Etc/GMT+7'
    )


@pytest.fixture
def albuquerque():
    """pvlib location for Albuquerque, NM."""
    return location.Location(
        35.0844,
        -106.6504,
        name='Albuquerque',
        altitude=1500,
        tx='Etc/GMT+7'
    )


@pytest.fixture
def summer_ghi(summer_times, albuquerque):
    """GHI and POA irradiance for a site in Albuquerque.

    The site has azimuth 180 and a 25 degree tilt.
    """
    clearsky = albuquerque.get_clearsky(summer_times)
    return clearsky['ghi']


def test_ghi_orientation_fixed(summer_ghi):
    """Clearsky GHI for summer months should has a FIXED Orientation"""
    assert system.orientation(
        summer_ghi,
        summer_ghi > 0,
        pd.Series(False, index=summer_ghi.index)
    ) is system.Orientation.FIXED
