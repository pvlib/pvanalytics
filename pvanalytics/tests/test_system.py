"""Tests for funcitons that identify system characteristics."""
import pytest
import pandas as pd
from pvlib import location
from pvanalytics import system


# Rought testing plan
#
# Generate several data sets (winter/summer only) using PVLib with
# different orientation and other characteristics and validate
# system.orientation by making sure the correct orientation is
# inferred.

# TODO Clearsky POA should be identifified as FIXED

# TODO Generate data using pvlib.tracking.SingleAxisTracker (TRACKING)

# TODO Generate power data from a PVSystem without a tracker


@pytest.fixture
def summer_times():
    """Ten-minute time stamps from May 1 through September 30, 2020 in GMT+7"""
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
    """Clearsky GHI for Summer, 2020 in Albuquerque, NM."""
    clearsky = albuquerque.get_clearsky(summer_times)
    return clearsky['ghi']


def test_ghi_orientation_fixed(summer_ghi):
    """Clearsky GHI for has a FIXED Orientation."""
    assert system.orientation(
        summer_ghi,
        summer_ghi > 0,
        pd.Series(False, index=summer_ghi.index)
    ) is system.Orientation.FIXED
