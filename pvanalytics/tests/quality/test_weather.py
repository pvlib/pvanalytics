"""Tests for functions in quality.weather"""
import pytest
import pandas as pd
import numpy as np
from pandas.util.testing import assert_series_equal
from pvanalytics.quality import weather


@pytest.fixture
def weather_data():
    """Weather data for use in tests."""
    output = pd.DataFrame(columns=['air_temperature', 'wind_speed',
                                   'relative_humidity',
                                   'extreme_temp_flag', 'extreme_wind_flag',
                                   'extreme_rh_flag'],
                          data=np.array([[-40, -5, -5, 0, 0, 0],
                                         [10, 10, 50, 1, 1, 1],
                                         [140, 55, 105, 0, 0, 0]]))
    dtypes = ['float64', 'float64', 'float64', 'bool', 'bool', 'bool']
    for (col, typ) in zip(output.columns, dtypes):
        output[col] = output[col].astype(typ)
    return output


def test_check_temperature_limits(weather_data):
    """Check that temperature values beyond limits are flagged."""
    assert_series_equal(
        weather.temperature_limits(weather_data['air_temperature']),
        weather_data['extreme_temp_flag'],
        check_names=False
    )


def test_relative_humidity_limits(weather_data):
    """Check that relative humidity values outside normal range are flagged."""
    assert_series_equal(
        weather.relative_humidity_limits(weather_data['relative_humidity']),
        weather_data['extreme_rh_flag'],
        check_names=False
    )


def test_wind_limits(weather_data):
    """Check that extremes in wind data are flagged."""
    assert_series_equal(
        weather.wind_limits(weather_data['wind_speed']),
        weather_data['extreme_wind_flag'],
        check_names=False
    )
