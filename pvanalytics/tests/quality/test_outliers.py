"""Tests for the quality.outliers module."""
import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal
from pvanalytics.quality import outliers
import pvlib
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from ..conftest import DATA_DIR


ac_power_test_file = DATA_DIR / "serf_east_15min_ac_power.csv"
weather_data_test_file = DATA_DIR / "serf_east_nsrdb_weather_data.csv"


def test_tukey_no_outliers():
    """If all the data has the same value there are no outliers"""
    data = pd.Series([1 for _ in range(10)])
    assert_series_equal(
        pd.Series([False for _ in range(10)]),
        outliers.tukey(data)
    )


def test_tukey_outlier_below():
    """tukey properly detects an outlier that is too low."""
    data = pd.Series([5, 9, 9, 8, 7, 1, 7, 8, 6, 8])
    assert_series_equal(
        pd.Series([False, False, False, False, False,
                  True, False, False, False, False]),
        outliers.tukey(data)
    )


def test_tukey_outlier_above():
    """tukey properly detects an outlier that is too high."""
    data = pd.Series([0, 1, 3, 2, 1, 3, 4, 1, 1, 9, 2])
    assert_series_equal(
        pd.Series([False, False, False, False, False,
                  False, False, False, False, True, False]),
        outliers.tukey(data)
    )


def test_tukey_lower_criteria():
    """With lower criteria the single small value is not an outlier."""
    data = pd.Series([5, 9, 9, 8, 7, 1, 7, 8, 6, 8])
    assert_series_equal(
        pd.Series([False for _ in range(len(data))]),
        outliers.tukey(data, k=3)
    )


def test_zscore_raise_nan_input():
    data = pd.Series([1, 0, -1, 0, np.nan, 1, -1, 10])

    with pytest.raises(ValueError):
        outliers.zscore(data, nan_policy='raise')


def test_zscore_invalid_nan_policy():
    data = pd.Series([1, 0, -1, 0, np.nan, 1, -1, 10])

    with pytest.raises(ValueError):
        outliers.zscore(data, nan_policy='incorrect_str')


def test_zscore_omit_nan_input():
    data = pd.Series([1, 0, -1, 0, np.nan, 1, -1, 10])
    assert_series_equal(
        pd.Series([False, False, False, False, False, False, False, True]),
        outliers.zscore(outliers.zscore(data, nan_policy='omit'))
    )


def test_zscore_all_same():
    """If all data is identical there are no outliers."""
    data = pd.Series([1 for _ in range(20)])
    np.seterr(invalid='ignore')
    assert_series_equal(
        pd.Series([False for _ in range(20)]),
        outliers.zscore(data)
    )
    np.seterr(invalid='warn')


def test_zscore_outlier_above():
    """Correctly idendifies an outlier above the mean."""
    data = pd.Series([1, 0, -1, 0, 1, -1, 10])
    assert_series_equal(
        pd.Series([False, False, False, False, False, False, True]),
        outliers.zscore(data)
    )


def test_zscore_outlier_below():
    """Correctly idendifies an outlier below the mean."""
    data = pd.Series([1, 0, -1, 0, 1, -1, -10])
    assert_series_equal(
        pd.Series([False, False, False, False, False, False, True]),
        outliers.zscore(data)
    )


def test_zscore_zmax():
    """Increasing zmax excludes outliers closest to the mean."""
    data = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 10])
    assert_series_equal(
        data[-2:],
        data[outliers.zscore(data)]
    )
    assert_series_equal(
        data[-1:],
        data[outliers.zscore(data, zmax=3)]
    )
    assert (~outliers.zscore(data, zmax=5)).all()


def test_hampel_all_same():
    """outliers.hampel identifies no outlier if all data is the same."""
    data = pd.Series(1, index=range(0, 50))
    assert_series_equal(
        outliers.hampel(data),
        pd.Series(False, index=range(0, 50))
    )


def test_hampel_one_outlier():
    """If all data is same but one value outliers.hampel should identify
    that value as an outlier."""
    np.random.seed(1000)
    data = pd.Series(np.random.uniform(0, 1, size=50))
    data.iloc[20] = 10
    expected = pd.Series(False, index=data.index)
    expected.iloc[20] = True
    assert_series_equal(
        outliers.hampel(data, window=11),
        expected
    )


def test_hampel_max_deviation():
    """Increasing max_deviation causes fewer values to be identified as
    outliers."""
    np.random.seed(1000)
    data = pd.Series(np.random.uniform(-1, 1, size=100))
    data.iloc[20] = -25
    data.iloc[40] = 15
    data.iloc[60] = 5

    expected = pd.Series(False, index=data.index)
    expected.iloc[[20, 40, 60]] = True

    assert_series_equal(
        data[outliers.hampel(data, window=11)],
        data[expected]
    )

    expected.iloc[60] = False
    assert_series_equal(
        data[outliers.hampel(data, window=11, max_deviation=15)],
        data[expected]
    )

    expected.iloc[40] = False
    assert_series_equal(
        data[outliers.hampel(data, window=11, max_deviation=16)],
        data[expected]
    )


def test_hampel_scale():
    np.random.seed(1000)
    data = pd.Series(np.random.uniform(-1, 1, size=100))
    data.iloc[20] = -25
    data.iloc[40] = 15
    data.iloc[60] = 5
    assert not all(outliers.hampel(data) == outliers.hampel(data, scale=0.1))


@pytest.fixture
def power_series():
    """Power series containing tz-aware datetime index and ac_power column."""
    power_df = pd.read_csv(ac_power_test_file, index_col=0, parse_dates=True)
    # Convert to UTC
    power_df.index = pd.to_datetime(power_df.index).tz_convert("UTC")
    power_series = power_df["ac_power"]
    return power_series


@pytest.fixture
def weather_df():
    """Weather data from NSRDB PSM3 api with UTC tz-aware datetime index."""
    weather_df = pd.read_csv(weather_data_test_file,
                             index_col=0, parse_dates=True)
    return weather_df


@pytest.fixture
def serf_east_metadata():
    """Metadata for serf east site."""
    lat = 39.742
    long = -105.1727
    azimuth = 158
    tilt = 45
    tracking = False
    capacity = 6
    return lat, long, azimuth, tilt, tracking, capacity


def test_run_pvwatts_data_checks_type_error(power_series, weather_df):
    """Test that TypeError for improper index input are raised.
    """
    # Test if TypeError is raised for non-indexed datetime
    with pytest.raises(TypeError,
                       match=('Power series must be a Pandas series with a ' +
                              'datetime index.')):
        outliers._run_pvwatts_data_checks(
            power_series.reset_index(), weather_df)
    with pytest.raises(TypeError,
                       match='Weather dataframe must have a datetime index.'):
        outliers._run_pvwatts_data_checks(
            power_series, weather_df.reset_index())


def test_run_pvwatts_data_checks_frequency(power_series, weather_df):
    """Test that power series matches the frequency of weather data.
    """
    resampled_power_series = outliers._run_pvwatts_data_checks(power_series,
                                                               weather_df)
    resampled_power_series_freq = pd.infer_freq(resampled_power_series.index)
    weather_df_freq = pd.infer_freq(weather_df.index)
    # Assert weather data equals resampled power series frequency
    assert weather_df_freq == resampled_power_series_freq


def test_run_pvwatts_model(serf_east_metadata, weather_df):
    """Test if power output is calculated for nontracking and tracking systems.
    """
    lat, long, azimuth, tilt, tracking, capacity = serf_east_metadata
    # Build out the PVWatts model
    solpos = pvlib.solarposition.get_solarposition(
        weather_df.index, lat, long)
    dni_extra = pvlib.irradiance.get_extra_radiation(
        weather_df.index)
    relative_airmass = pvlib.atmosphere.get_relative_airmass(solpos.zenith)
    temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    # Run pvwatts model for tracking systems
    tracking_predicted_power = outliers._run_pvwatts_model(
        tilt=tilt,
        azimuth=azimuth,
        dc_capacity=capacity,
        solar_zenith=solpos.zenith,
        solar_azimuth=solpos.azimuth,
        dni=weather_df['DNI'],
        dhi=weather_df['DHI'],
        ghi=weather_df['GHI'],
        dni_extra=dni_extra,
        relative_airmass=relative_airmass,
        temperature=weather_df['Temperature'],
        wind_speed=weather_df['Wind Speed'],
        temperature_model_parameters=temp_params,
        temperature_coefficient=-0.0047,
        tracking=True)
    # Run pvwatts model for nontracking systems
    nontracking_predicted_power = outliers._run_pvwatts_model(
        tilt=tilt,
        azimuth=azimuth,
        dc_capacity=capacity,
        solar_zenith=solpos.zenith,
        solar_azimuth=solpos.azimuth,
        dni=weather_df['DNI'],
        dhi=weather_df['DHI'],
        ghi=weather_df['GHI'],
        dni_extra=dni_extra,
        relative_airmass=relative_airmass,
        temperature=weather_df['Temperature'],
        wind_speed=weather_df['Wind Speed'],
        temperature_model_parameters=temp_params,
        temperature_coefficient=-0.0047,
        tracking=tracking)

    # Assert that the predicted power output matches the length of weather_df
    assert len(tracking_predicted_power) == len(weather_df)
    assert len(nontracking_predicted_power) == len(weather_df)
    # Assert that the datetime index are the same
    assert tracking_predicted_power.index.equals(weather_df.index)
    assert nontracking_predicted_power.index.equals(weather_df.index)
    # Assert that nontracking and tracking power output are not equal
    assert not tracking_predicted_power.equals(nontracking_predicted_power)


def test_calc_abs_percent_diff():
    datetime_index = pd.date_range(
        start="2024-01-01", periods=5, freq="D", tz="UTC")
    actual_series = pd.Series([1, 1, 2, 4, 1], index=datetime_index)
    predicted_series = pd.Series([1, 3, -2, 1, 4], index=datetime_index)
    expected_abs_pct_diff = pd.Series([0, 100, float("inf"), 120, 120],
                                      index=datetime_index)
    actual_abs_pct_diff = outliers._calc_abs_percent_diff(actual_series,
                                                          predicted_series)
    # Asset that expected values are the same as the actual value
    assert (actual_abs_pct_diff == expected_abs_pct_diff).all()


def test_pvwatts_vs_actual_abs_percent_diff(serf_east_metadata,
                                            power_series, weather_df):
    """Test if the correct daily frequency is returned.
    """
    lat, long, azimuth, tilt, tracking, _ = serf_east_metadata
    # Assume there's no capacity given
    pct_diff_series = outliers.pvwatts_vs_actual_abs_percent_diff(
        power_series, lat, long, tilt, azimuth,
        tracking, weather_df, dc_capacity=None)
    pct_diff_series_freq = pd.infer_freq(pct_diff_series.index)
    # Assert that the series is returned with datetime index
    assert isinstance(pct_diff_series.index, pd.DatetimeIndex)
    # Assert that the series has daily frequency
    assert pct_diff_series_freq == "D"


def test_flag_irregular_power_days():
    """Tests if returned series is boolean and returns result as expected.
    """
    datetime_index = pd.date_range(
        start="2024-01-01", periods=5, freq="D", tz="UTC")
    pct_diff_series = pd.Series([0, 80, 35, 61, 11], index=datetime_index)
    expected_irregular_day_series = pd.Series(
        [False, True, False, True, False], index=datetime_index)
    actual_irregular_day_series = outliers.flag_irregular_power_days(
        pct_diff_series, pct_threshold=50)
    # Asset that expected values are the same as the actual value
    assert (actual_irregular_day_series == expected_irregular_day_series).all()
