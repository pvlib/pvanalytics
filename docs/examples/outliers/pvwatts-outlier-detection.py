"""
PVWatts Outlier Detection
=========================

Identifying outliers in time series using
PVWatts outlier detection.
"""

# %%
# Identifying and removing outliers from PV sensor time series
# data allows for more accurate data analysis.
# In this example, we demonstrate how to use
# :py:func:`pvanalytics.quality.outliers.pvwatts_vs_actual_abs_percent_diff`
# along with
# :py:func:`pvanalytics.quality.outliers.flag_irregular_power_days`
# to identify abnormal daily behavior in a time series by measuring
# a data stream's daily performance against a PVWatts model.

import pvanalytics
import pvanalytics.quality.outliers as outliers
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

# %%
# First, we import an AC power data stream from the SERF East site located at
# NREL. This data set is publicly available via the PVDAQ database in the DOE
# Open Energy Data Initiative (OEDI)
# (https://data.openei.org/submissions/4568), under system ID 50.
# This data is timezone-localized.

pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
ac_power_file_1 = pvanalytics_dir / 'data' / 'serf_east_15min_ac_power.csv'
ac_power_data = pd.read_csv(ac_power_file_1, index_col=0, parse_dates=True)
ac_power_data.index = pd.to_datetime(ac_power_data.index).tz_convert("UTC")
ac_power_series = ac_power_data["ac_power"]
print(ac_power_series.head(10))

# %%
# Second, we import weather data from NSRDB PSM3 api.
# We specifically pulled ambient temperature, dhi, dni, and wind speed values
# at the SERF East site and for the entire duration of the AC power
# measurement. By default, the pulled weather data is in UTC
# tz-aware datetime. Users can sign up for an api key at NREL Developer
# Network https://developer.nrel.gov/signup/ and use pvlib.iotools.get_psm3
# to query the api. See pvlib documentaton for the full
# list of fields in NSRDB
# https://pvlib-python.readthedocs.io/en/v0.9.0/generated/pvlib.iotools.get_psm3.html
pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
weather_file_1 = pvanalytics_dir / 'data' / 'serf_east_nsrdb_weather_data.csv'
weather_data = pd.read_csv(weather_file_1, index_col=0, parse_dates=True)
print(weather_data.head(10))

# %%
# We then get the metadata for SERF East site and
# use:py:func:`pvanalytics.quality.outliers.pvwatts_vs_actual_abs_percent_diff`
# to get the absolute percent difference between the actual and predicted
# time series. The predicted AC power time series
# is modeled from PVWatts with NSRDB data and site metadata.

lat = 39.742
long = -105.1727
azimuth = 158
tilt = 45
tracking = False
capacity = ac_power_series.max()
# Get absolute percent difference
abs_percent_diff_series = outliers.pvwatts_vs_actual_abs_percent_diff(
    power_time_series=ac_power_series,
    lat=lat,
    long=long,
    tilt=tilt,
    azimuth=azimuth,
    tracking=tracking,
    nsrdb_weather_df=weather_data,
    dc_capacity=capacity)
print(abs_percent_diff_series.tail(10))

# %%
# We then flag the days where the absolute percent difference exceeds
# the set precent threshold of 50%. We can use
# :py:func:`pvanalytics.quality.outliers.flag_irregular_power_days`
# for this flagging the irregular days.
# The actual and predicted daily time series with detected anomalous days
# are then plotted.

irregular_day_series = outliers.flag_irregular_power_days(
    abs_percent_diff_series, pct_threshold=50)
print(irregular_day_series.tail(10))

# Plot results
abs_percent_diff_series.plot()
abs_percent_diff_series.loc[irregular_day_series].plot(ls='', marker='o')
plt.legend(labels=["Absolute Percent Difference", "Detected Outlier"])
plt.xlabel("Date")
plt.ylabel("Absolute Percent Difference")
plt.tight_layout()
plt.show()
