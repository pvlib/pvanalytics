"""Functions for identifying and labeling outliers."""
import pandas as pd
from scipy import stats
from statsmodels import robust
import pvlib
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS


def tukey(data, k=1.5):
    r"""Identify outliers based on the interquartile range.

    A value `x` is considered an outlier if it does *not* satisfy the
    following condition

    .. math::
        Q_1 - k(Q_3 - Q_1) \le x \le Q_3 + k(Q_3 - Q_1)

    where :math:`Q_1` is the value of the first quartile and
    :math:`Q_3` is the value of the third quartile.

    Parameters
    ----------
    data : Series
        The data in which to find outliers.
    k : float, default 1.5
        Multiplier of the interquartile range. A larger value will be more
        permissive of values that are far from the median.

    Returns
    -------
    Series
        A series of booleans with True for each value that is an
        outlier.

    """
    first_quartile = data.quantile(0.25)
    third_quartile = data.quantile(0.75)
    iqr = third_quartile - first_quartile
    return ((data < (first_quartile - k*iqr))
            | (data > (third_quartile + k*iqr)))


def zscore(data, zmax=1.5, nan_policy='raise'):
    """Identify outliers using the z-score.

    Points with z-score greater than `zmax` are considered as outliers.

    Parameters
    ----------
    data : Series
        A series of numeric values in which to find outliers.
    zmax : float
        Upper limit of the absolute values of the z-score.
    nan_policy : {'raise', 'omit'}, default 'raise'
        Define how to handle NaNs in the input series.
        If 'raise', a ValueError is raised when `data` contains NaNs.
        If 'omit', NaNs are ignored and False is returned at indices that
        contained NaN in `data`.

    Returns
    -------
    Series
        A series of booleans with True for each value that is an
        outlier.

    """
    nan_mask = pd.Series([False] * len(data),
                         index=data.index)

    if data.hasnans:
        if nan_policy == 'raise':
            raise ValueError("The input contains nan values.")
        elif nan_policy == 'omit':
            nan_mask = data.isna()
        else:
            raise ValueError(f"Unnexpected value ({nan_policy}) passed to "
                             "nan_policy. Expected 'raise' or 'omit'.")

    is_outlier = pd.Series(False, index=data.index)
    is_outlier.loc[~nan_mask] = abs(stats.zscore(data[~nan_mask])) > zmax
    return is_outlier


def hampel(data, window=5, max_deviation=3.0, scale=None):
    r"""Identify outliers by the Hampel identifier.

    The Hampel identifier is computed according to [1]_.

    Parameters
    ----------
    data : Series
        The data in which to find outliers.
    window : int or offset, default 5
        The size of the rolling window used to compute the Hampel
        identifier.
    max_deviation : float, default 3.0
        Any value with a Hampel identifier > `max_deviation` standard
        deviations from the median is considered an outlier.
    scale : float, optional
        Scale factor used to estimate the standard deviation as
        :math:`MAD / scale`. If `scale=None` (default), then the scale
        factor is taken to be ``scipy.stats.norm.ppf(3/4.)`` (approx. 0.6745),
        and :math:`MAD / scale` approximates the standard deviation
        of Gaussian distributed data.

    Returns
    -------
    Series
        True for each value that is an outlier according to its Hampel
        identifier.

    References
    ----------
    .. [1] Pearson, R.K., Neuvo, Y., Astola, J. et al. Generalized
       Hampel Filters. EURASIP J. Adv. Signal Process. 2016, 87
       (2016). https://doi.org/10.1186/s13634-016-0383-6

    """
    median = data.rolling(window=window, center=True).median()
    deviation = abs(data - median)
    kwargs = {}
    if scale is not None:
        kwargs = {'c': scale}
    mad = data.rolling(window=window, center=True).apply(
        robust.scale.mad,
        kwargs=kwargs
    )
    return deviation > max_deviation * mad


def _run_pvwatts_data_checks(power_series, nsrdb_weather_df):
    """Check that the passed parameters can be run through the function.

    This includes checking the passed time series to ensure it has a
    datetime index, and that its frequency matches with nsrdb weather
    data frequency.

    Parameters
    ----------
    power_series : Pandas series with datetime index.
        Time series of a PV power data stream with UTC datetime tz-aware index.
    nsrdb_weather_df : Pandas dataframe with datetime index
        A pandas dataframe containing NSRDB PSM3 weather 'Temperature', 'DHI',
        'DNI', and 'Wind Speed' columns with datetime index. The minimum
        and maximum datetime matches the minimum and maximum datetime of the
        actual data.

    Returns
    -------
    daily_series : Pandas series with datetime index.
        Daily time series of a PV power data stream. This series represents
        the summed daily values of the particular data stream.
    """
    # Check that the series and df has a datetime index
    if not isinstance(power_series.index, pd.DatetimeIndex):
        raise TypeError('Power series must be a Pandas series with a ' +
                        'datetime index.')
    if not isinstance(nsrdb_weather_df.index, pd.DatetimeIndex):
        raise TypeError('Weather dataframe must have a datetime index.')
    # Get the frequency of weather data
    nsrdb_weather_df = pd.infer_freq(nsrdb_weather_df.index)
    # Resample power series to same frequency as weather data
    power_series = power_series.resample(nsrdb_weather_df).mean()
    return power_series


def _run_pvwatts_model(tilt, azimuth, dc_capacity, solar_zenith,
                       solar_azimuth, dni, dhi, ghi, dni_extra,
                       relative_airmass, temperature, wind_speed,
                       temperature_model_parameters,
                       temperature_coefficient, tracking):
    r"""Run the PVWatts model with NSRDB data across the time period as inputs.

    Parameters
    ----------
    tilt : Float
        Tilt angle of the data stream in degrees.
    azimuth : Float
        Azimuth angle of the data stream in degrees.
    dc_capacity : Float
        DC capacity of the the data stream.
    solar_zenith : Series
        Solar zenith angle in degrees
    solar_azimuth :
        Azimuth angle of the sun in degrees East of North
    dni : Series
        Direct normal irradiance in :math:`W/m^2`
    dhi : Series
        Diffuse horizontal irradiance in :math:`W/m^2`
    ghi : Series
        Global horizontal irradiance in :math:`W/m^2`
    dni_extra : Series
        Extraterrestrial normal irradiance in :math:`W/m^2`
    relative_airmass : Series
        Relative airmass (not adjusted for pressure)
    temperature : Series
        Ambient dry bulb temperature in C.
    wind_speed : Series
        Wind speed at a height of 10 meters in :math:`m/s`.
    temperature_model_parameters : Dict
        Parameters for the cell temperature model.
    temperature_coefficient : Float
        Temperature coefficient of DC power. [1/C]
    tracking : Boolean
        True if tracking. Otherwise, False if not tracking.

    Returns
    -------
    pdc : Series
        Pandas series of DC power output modeled from PVWatts with
        datetime index.
    """
    if tracking:
        tracker_angles = pvlib.tracking.singleaxis(solar_zenith,
                                                   solar_azimuth,
                                                   axis_tilt=tilt,
                                                   axis_azimuth=azimuth,
                                                   backtrack=True,
                                                   gcr=0.4, max_angle=60)
        surface_tilt = tracker_angles['surface_tilt']
        surface_azimuth = tracker_angles['surface_azimuth']
    else:
        surface_tilt = tilt
        surface_azimuth = azimuth

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt, surface_azimuth,
        solar_zenith,
        solar_azimuth,
        dni, ghi, dhi,
        dni_extra=dni_extra,
        airmass=relative_airmass,
        albedo=0.2,
        model='perez'
    )

    aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth,
                               solar_zenith, solar_azimuth)
    # Run IAM model
    iam = pvlib.iam.physical(aoi, n=1.5)
    # Apply IAM to direct POA component only
    poa_transmitted = poa['poa_direct'] * iam + poa['poa_diffuse']
    temp_cell = pvlib.temperature.sapm_cell(
        poa['poa_global'],
        temperature,
        wind_speed,
        **temperature_model_parameters
    )
    pdc = pvlib.pvsystem.pvwatts_dc(
        poa_transmitted,
        temp_cell,
        dc_capacity,
        temperature_coefficient
    )
    return pdc


def _calc_abs_percent_diff(actual_series, expected_series):
    """Gets the absolute percent difference for actual and predicted series.

    Parameters
    ----------
    actual_series : Pandas series
        Series of actual, measured data with matching index as expected_series.
    expected_series : Pandas series
        Series of predicted data with matching index as actual_series.

    Returns
    -------
    abs_percent_diff_series : Pandas series
        Absolute percent difference between the actual and predicted data.
    """
    diff_val = actual_series - expected_series
    mean_val = (actual_series + expected_series) / 2
    abs_percent_diff_series = abs(diff_val / mean_val) * 100
    return abs_percent_diff_series


def pvwatts_vs_actual_abs_percent_diff(power_time_series, lat, long, tilt,
                                       azimuth, tracking, nsrdb_weather_df,
                                       dc_capacity=None):
    """Gets the absolute percent difference for actual and PVWatts time series.

    Compares the actual daily power time series data with PVWatts
    modeled power time series by calculating their absolute percent
    differences.

    Parameters
    ----------
    power_time_series : Pandas series with datetime index.
        Time series of a PV power data stream with UTC tz-datetime aware index.
    lat : Float
        Latitude value of the data stream.
    long : FLoat
        Longitude value of the data stream.
    tilt : Float
        Tilt angle of the data stream in degrees.
    azimuth : Float
        Azimuth angle of the data stream in degrees.
    tracking : Boolean
        True if tracking. Otherwise, False if not tracking.
    nsrdb_weather_df : Pandas dataframe with datetime index
        A pandas dataframe containing NSRDB PSM3 weather 'Temperature', 'DHI',
        'DNI', and 'Wind Speed' columns with datetime index. The minimum
        and maximum datetime matches the minimum and maximum datetime of the
        actual data.
    dc_capacity : None or Float
        DC capacity of the the data stream. If the dc capacity is not
        known, set as None so that it can be calculated by finding the 95%
        percentile of the passed time series.
        Defaulted to None.

    Returns
    -------
    abs_percent_diff_series : Pandas series with datetime index.
        Absolute percent difference between the actual and predicted data
        with UTC tz-datetime aware index.

    """
    # Run data check on time series data to make sure index is datetime
    # and it matches nsrdb weather data frequency
    actual_power_ts = _run_pvwatts_data_checks(power_time_series,
                                               nsrdb_weather_df)
    # If dc_capacity is None, calculate the 95% percentile
    if dc_capacity is None:
        dc_capacity = actual_power_ts.quantile(0.95)

    # Build out the PVWatts model
    solpos = pvlib.solarposition.get_solarposition(
        nsrdb_weather_df.index, lat, long)
    dni_extra = pvlib.irradiance.get_extra_radiation(
        nsrdb_weather_df.index)
    relative_airmass = pvlib.atmosphere.get_relative_airmass(solpos.zenith)
    temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    # Run PVWatts model
    predicted_power_ts = _run_pvwatts_model(
        tilt=tilt,
        azimuth=azimuth,
        dc_capacity=dc_capacity,
        solar_zenith=solpos.zenith,
        solar_azimuth=solpos.azimuth,
        dni=nsrdb_weather_df['DNI'],
        dhi=nsrdb_weather_df['DHI'],
        ghi=nsrdb_weather_df['GHI'],
        dni_extra=dni_extra,
        relative_airmass=relative_airmass,
        temperature=nsrdb_weather_df['Temperature'],
        wind_speed=nsrdb_weather_df['Wind Speed'],
        temperature_model_parameters=temp_params,
        temperature_coefficient=-0.0047,
        tracking=tracking)

    # Combine actual and predicted power data
    predicted_power_df = predicted_power_ts.to_frame(
        'predicted_power').rename_axis("datetime")
    actual_power_df = actual_power_ts.to_frame(
        'actual_power').rename_axis("datetime")
    compare_power_df = pd.merge(actual_power_df, predicted_power_df,
                                left_index=True, right_index=True)
    # Resample to daily frequency and sum to get total daily output
    compare_power_df = compare_power_df.resample("D").sum()
    # Get percent difference
    abs_percent_diff_series = _calc_abs_percent_diff(
        compare_power_df["actual_power"], compare_power_df["predicted_power"])

    return abs_percent_diff_series


def flag_irregular_power_days(percent_diff_series, pct_threshold=50):
    """Flags the days with abnormal power output behavior.

    Days where the difference over the specified percent difference threshold
    are flagged as irregular.

    Parameters
    ----------
    percent_diff_series : Pandas series with datetime index.
        Percent difference between the actual and predicted data
        with UTC tz-datetime aware index.
    pct_threshold : Float
        Percent difference threshold for flagging data as anomalies.
        Defaulted to 50.

    Returns
    -------
    irregular_day_series : Pandas series with datetime index.
       Boolean values that flag if the day is an anomaly if True. Otherwise,
       False if the data stream is producing the output as expected.

    """
    irregular_day_series = percent_diff_series > pct_threshold
    return irregular_day_series
