"""Quality control functions for inverter DC capacity time series data.
"""

import pandas as pd
import os
import pvlib
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from matplotlib import pyplot as plt
from psm3 import geohash, nsrdb_weather


def _run_data_checks(series):
    """
    Check that the passed parameters can be run through the function.
    This includes checking the passed time series to ensure it has a
    datetime index, and that it is a daily sampled time series. If time
    series data is not daily sampled, then automatically resample it to daily.

    Parameters
    ----------
    series : Pandas series with datetime index.
        Time series of a PV DC data stream.

    Returns
    -------
    daily_series : Pandas series with datetime index.
        Daily time series of a PV dc data stream. This series represents
        the summed daily values of the particular data stream.
    """
    # Check that the time series has a datetime index, and consists of numeric
    # values
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError('Must be a Pandas series with a datetime index.')
    # Check that the time series is sampled on a daily basis. If not,
    # automatically resample the data to daily
    if series.index.to_series().diff().value_counts().idxmax().days != 1:
        print('Input data is not daily frequency. ' +
              'Resampling to daily frequency with .sum()')
        daily_series = series.resample("D").sum()
    return daily_series


def run_pvwatts_model(tilt, azimuth, dc_capacity, dc_inverter_limit,
                      solar_zenith, solar_azimuth, dni, dhi, ghi, dni_extra,
                      relative_airmass, temperature, wind_speed,
                      temperature_model_parameters,
                      temperature_coefficient, tracking):
    """
    Run the PVWatts model using NSRDB data across the time period as inputs.

    Parameters
    ----------
    tilt : Float
        Tilt angle of site in degrees.
    azimuth : Float
        Azimuth angle of site in degrees.
    dc_capacity : Float
        DC capacity of the inverter in kW.
    dc_inverter_limit : Float
        Maximum DC capacity for the inverter in kW.
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
        Pandas series of DC power output in kW modeled from PVWatts with
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


def build_pvwatts_model(lat, long, tilt, azimuth, power, tracking,
                        nsrdb_email, nsrdb_api_key, min_measured_date,
                        max_measured_date, pvwatts_output_path):
    """
    Builds the PVWatts model by getting the nsrdb weather data and modeling
    the system with its metadata and those weather metadata.

    Parameters
    ----------
    lat : Float
        Latitude value of site.
    long : FLoat
        Longitude value of site.
    tilt : Float
        Tilt angle of site in degrees.
    azimuth : Float
        Azimuth angle of site in degrees.
    power : Float
        DC capacity of the inverter in kW.
    tracking : Boolean
        True if tracking. Otherwise, False if not tracking.
    nsrdb_email : Str
        NSRDB email to pull nsrdb weather data. 
    nsrdb_api_key : Str
        NSRDB api key to pull nsrdb weather data. 
    min_measured_date : Pandas datetime.
        Minimum measured datetime of time series.
    max_measured_date : Pandas datetime.
        Maximum measured datetime of time series.
    pvwatts_output_path : None or Str
        Path to save PVWatts output data. If None, no data will be saved.

    Returns
    -------
    master_df : Pandas dataframe with datetime index.
        A pandas dataframe containing columns 'actual_dc_output_kW',
        'predicted_dc_output_kW', 'percent_difference', and 'anomalous'
        columns and datetime index.
    """
    # Get the geohash associated with the site
    geohash_val = geohash(lat, long, precision=6)
    # Pull the site's associated NSRDB data
    master_weather_df = pd.DataFrame()
    for year in range(min_measured_date.year, max_measured_date.year):
        for try_time in range(0, 3):
            try:
                df = nsrdb_weather(geohash_val,
                                   year, nsrdb_email, nsrdb_api_key,
                                   interval=30,
                                   attributes={'Temperature': 'temp_air',
                                               'DHI': 'dhi',
                                               'DNI': 'dni',
                                               'GHI': 'ghi',
                                               'Wind Speed': 'wind_speed'})
                master_weather_df = pd.concat([master_weather_df, df])
                break
            except:
                pass

    # Build out the PVWatts model
    solpos = pvlib.solarposition.get_solarposition(master_weather_df.index,
                                                   lat, long)
    dni_extra = pvlib.irradiance.get_extra_radiation(
        master_weather_df.index)
    relative_airmass = pvlib.atmosphere.get_relative_airmass(solpos.zenith)
    temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    pdc = run_pvwatts_model(tilt=tilt,
                            azimuth=azimuth,
                            dc_capacity=power,
                            dc_inverter_limit=power * 1.5,
                            solar_zenith=solpos.zenith,
                            solar_azimuth=solpos.azimuth,
                            dni=master_weather_df['DNI'],
                            dhi=master_weather_df['DHI'],
                            ghi=master_weather_df['GHI'],
                            dni_extra=dni_extra,
                            relative_airmass=relative_airmass,
                            temperature=master_weather_df['Temperature'],
                            wind_speed=master_weather_df['Wind Speed'],
                            temperature_model_parameters=temp_params,
                            temperature_coefficient=-0.0047,
                            tracking=tracking)

    # Plot PDC against real
    pdc.plot()
    # Plot real
    plt.show()
    plt.close()
    pdc.name = 'output_kW'
    if pvwatts_output_path is not None:
        # Write the results to the designated output path
        pdc.to_csv(os.path.join(pvwatts_output_path + ".csv"))

    return pdc


def get_anomalous_days(dc_time_series, lat, long, tilt, azimuth, tracking,
                       nsrdb_email, nsrdb_api_key,
                       pct_threshold=0.50,
                       dc_capacity=None,
                       pvwatts_output_path=None):
    """
    Compares the actual daily inverter DC time series data with PVWatts model
    DC time series output by calculating their percent deference. Days where
    the difference over the specified percent difference threshold are flagged
    as anomalous.

    Parameters
    ----------
    dc_time_series : Pandas series with datetime index.
        Time series of a PV DC data stream. The units for DC should be kW.
    lat : Float
        Latitude value of site.
    long : FLoat
        Longitude value of site.
    tilt : Float
        Tilt angle of site in degrees.
    azimuth : Float
        Azimuth angle of site in degrees.
    tracking : Boolean
        True if tracking. Otherwise, False if not tracking.
    nsrdb_email : Str
        NSRDB email to pull nsrdb weather data. 
    nsrdb_api_key : Str
        NSRDB api key to pull nsrdb weather data. 
    pct_threshold : Float
        Percent difference threshold for flagging data as anomalies.
        Defaulted to 0.50.
    dc_capacity : None or Float
        DC capacity of the inverter in kW. If the inverter dc capacity is not
        known, set as None so that it can be calculated by finding the 95%
        percentile of the passed time series.
        Defaulted to None.
    pvwatts_output_path : None or Str
        Path to save PVWatts output data. If None, no data will be saved.
        Defaulted to None.

    Returns
    -------
    master_df : Pandas dataframe with datetime index.
        A pandas dataframe containing columns 'actual_dc_output_kW',
        'predicted_dc_output_kW', 'percent_difference', and 'anomalous'
        columns and datetime index.

    """
    # Run data check on time series data to make sure index is datetime
    # and data is daily sampled
    actual_dc_ts = _run_data_checks(dc_time_series)
    # If dc_capacity is None, calculate the 95% percentile
    if dc_capacity is None:
        dc_capacity = actual_dc_ts.quantile(0.95)
    # Get min and max date time range from datetime index
    min_measured_date = pd.to_datetime(actual_dc_ts.index.min())
    max_measured_date = pd.to_datetime(actual_dc_ts.index.max())
    # Get predicted dc time series data from PVWatts
    predicted_dc_ts = build_pvwatts_model(
        lat=lat,
        long=long,
        tilt=tilt,
        azimuth=azimuth,
        power=dc_capacity,
        tracking=tracking,
        nsrdb_email=nsrdb_email,
        nsrdb_api_key=nsrdb_api_key,
        min_measured_date=min_measured_date,
        max_measured_date=max_measured_date,
        pvwatts_output_path=pvwatts_output_path)
    # Resample PVWatts predicted DC time series to daily frequency
    daily_predicted_dc_ts = predicted_dc_ts.resample('D').sum()
    # Combine actual and predicted dc data
    master_df = pd.DataFrame(
        index=actual_dc_ts.index,
        data={'actual_dc_output_kW': actual_dc_ts,
              'predicted_dc_output_kW': daily_predicted_dc_ts})
    master_df['predicted_dc_output_kW'] = daily_predicted_dc_ts
    # Get percent difference
    master_df['percent_difference'] = abs(
        master_df['actual_dc_output_kW'] - master_df['predicted_dc_output_kW']
    ) / master_df['predicted_dc_output_kW']
    # Mark as anomalous (True) if percent difference passes the threshold
    # Otherwise, False
    master_df['anomalous'] = master_df['percent_difference'] > pct_threshold

    return master_df
