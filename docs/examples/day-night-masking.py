"""
Day-Night Masking
=================

Masking day-night periods using the PVAnalytics daytime module.
"""

# %%
# Identifying and masking day-night periods in an AC power time series or
# irradiance time series can aid in future data analysis, such as detecting
# if a time series has daylight savings time or time shifts.

import pvanalytics
from pvanalytics.features.daytime import power_or_irradiance
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import pvlib
import numpy as np

# %%
# First, read in the 1-minute sampled AC power time series data, taken
# from the SERF East installation on the NREL campus.
# This sample is provided from the NREL PVDAQ database, and contains
# a column representing an AC power data stream.

pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
ac_power_file = pvanalytics_dir / 'data' / 'serf_east_1min_ac_power.csv'
data = pd.read_csv(ac_power_file, index_col=0, parse_dates=True)
data = data.sort_index()
# This is the known frequency of the time series. You may need to infer
# the frequency or set the frequency with your AC power time series.
freq = "1T"
# These are the latitude-longitude coordinates associated with the
# SERF East system.
latitude = 39.742
longitude = -105.173
# Plot the time series.
data['ac_power__752'].plot()
plt.xticks(rotation=20)
plt.xlabel("Date")
plt.ylabel("AC Power (kW)")
plt.tight_layout()
plt.show()

# %%
# The pvlib.solarposition.sun_rise_set_transit_spa function is
# used to get sunrise and sunset times for each day at the site location, and
# the daytime mask is calculated based on these times. Data associated with
# day time periods is labeled as True, and data associated with nighttime
# periods is labeled as False.
sunrise_sunset_df = pvlib.solarposition.sun_rise_set_transit_spa(data.index,
                                                                 latitude,
                                                                 longitude,
                                                                 how='numpy',
                                                                 delta_t=67.0)
data['sunrise_time'] = sunrise_sunset_df['sunrise']
data['sunset_time'] = sunrise_sunset_df['sunset']

data['daytime_mask'] = True
data.loc[(data.index < data.sunrise_time) |
         (data.index > data.sunset_time), "daytime_mask"] = False

# %%
# Set all negative values in the AC power time series to 0.
data.loc[data['ac_power__752'] < 0, 'ac_power__752'] = 0

# %%
# Now, use :py:func:`pvanalytics.features.daytime.power_or_irradiance`
# to identify day periods in the time series. Re-plot the data
# subset with this mask.
predicted_day_night_mask = power_or_irradiance(series=data['ac_power__752'],
                                               freq=freq)
data['ac_power__752'].plot()
data.loc[predicted_day_night_mask, 'ac_power__752'].plot(ls='', marker='o')
data.loc[~predicted_day_night_mask, 'ac_power__752'].plot(ls='', marker='o')
plt.legend(labels=["AC Power", "Daytime", "Nighttime"])
plt.xticks(rotation=20)
plt.xlabel("Date")
plt.ylabel("AC Power (kW)")
plt.tight_layout()
plt.show()

# %%
# Compare the predicted mask to the ground-truth mask, to get the model
# accuracy. Also, compare sunrise and sunset times for the predicted mask
# compared to the ground truth sunrise and sunset times.
acc = 100 * np.sum(np.equal(data.daytime_mask,
                            predicted_day_night_mask))/len(data.daytime_mask)
print("Overall model prediction accuracy: " + str(round(acc, 2)) + "%")

# Get predicted sunrise and sunset times
predicted_day_night_df = pd.concat([pd.Series(
    predicted_day_night_mask.index.date, predicted_day_night_mask.index,
    name="date"),
    pd.Series(predicted_day_night_mask.index, predicted_day_night_mask.index,
              name="datetime"), predicted_day_night_mask], axis=1)
sunrise = predicted_day_night_df[predicted_day_night_df['ac_power__752']]\
    .groupby(['date', 'ac_power__752'])['datetime'].min()
sunset = predicted_day_night_df[predicted_day_night_df['ac_power__752']]\
    .groupby(['date', 'ac_power__752'])['datetime'].max()
predicted_sunrise_sunset = pd.concat([sunrise.rename('predicted_sunrise'),
                                      sunset.rename('predicted_sunset')],
                                     axis=1).reset_index(drop=True)
predicted_sunrise_sunset = predicted_sunrise_sunset.set_index(
    predicted_sunrise_sunset.predicted_sunrise.dt.date)
gt_sunrise_sunset = sunrise_sunset_df[['sunrise',
                                       'sunset']].\
    drop_duplicates().reset_index(drop=True)
gt_sunrise_sunset = gt_sunrise_sunset.set_index(
    gt_sunrise_sunset.sunrise.dt.date)

# Compare predicted sunrise and sunset times to ground truth sunrise
# and sunset times
sunrise_sunset_comparison_df = pd.concat([predicted_sunrise_sunset,
                                          gt_sunrise_sunset], axis=1)
print("Comparing sunrise times:")
print(sunrise_sunset_comparison_df[['sunrise', 'predicted_sunrise']])
print("Comparing sunset times:")
print(sunrise_sunset_comparison_df[['sunset', 'predicted_sunset']])

# %%
# Now we repeat the above process with 15-minute sampled AC power time series
# data, taken from the SERF East installation on the NREL campus. This data
# is derived from the 1-minute data above, except it has been resampled to
# 15-minute right-aligned mean data.
ac_power_file = pvanalytics_dir / 'data' / 'serf_east_15min_ac_power.csv'
data = pd.read_csv(ac_power_file, index_col=0, parse_dates=True)
data = data.sort_index()
# This is the known frequency of the time series. You may need to infer
# the frequency or set the frequency with your AC power time series.
freq = "15T"
# Plot the time series.
data['ac_power__752'].plot()
plt.xticks(rotation=20)
plt.xlabel("Date")
plt.ylabel("AC Power (kW)")
plt.tight_layout()
plt.show()

# %%
# The pvlib.solarposition.sun_rise_set_transit_spa function is
# used to get sunrise and sunset times for each day at the site location, and
# the daytime mask is calculated based on these times. Data associated with
# day time periods is labeled as True, and data associated with nighttime
# periods is labeled as False.
sunrise_sunset_df = pvlib.solarposition.sun_rise_set_transit_spa(data.index,
                                                                 latitude,
                                                                 longitude,
                                                                 how='numpy',
                                                                 delta_t=67.0)
data['sunrise_time'] = sunrise_sunset_df['sunrise']
data['sunset_time'] = sunrise_sunset_df['sunset']

data['daytime_mask'] = True
data.loc[(data.index < data.sunrise_time) |
         (data.index > data.sunset_time), "daytime_mask"] = False

# %%
# Set all negative values in the AC power time series to 0.
data.loc[data['ac_power__752'] < 0, 'ac_power__752'] = 0

# %%
# Now, use :py:func:`pvanalytics.features.daytime.power_or_irradiance`
# to identify day periods in the time series. Re-plot the data
# subset with this mask.
predicted_day_night_mask = power_or_irradiance(series=data['ac_power__752'],
                                               freq=freq)
data['ac_power__752'].plot()
data.loc[predicted_day_night_mask, 'ac_power__752'].plot(ls='', marker='o')
data.loc[~predicted_day_night_mask, 'ac_power__752'].plot(ls='', marker='o')
plt.legend(labels=["AC Power", "Daytime", "Nighttime"])
plt.xticks(rotation=20)
plt.xlabel("Date")
plt.ylabel("AC Power (kW)")
plt.tight_layout()
plt.show()

# %%
# Compare the predicted mask to the ground-truth mask, to get the model
# accuracy. Also, compare sunrise and sunset times for the predicted mask
# compared to the ground truth sunrise and sunset times.
acc = 100 * np.sum(np.equal(data.daytime_mask,
                            predicted_day_night_mask))/len(data.daytime_mask)
print("Overall model prediction accuracy: " + str(round(acc, 2)) + "%")

# Get predicted sunrise and sunset times
predicted_day_night_df = pd.concat([pd.Series(
    predicted_day_night_mask.index.date, predicted_day_night_mask.index,
    name="date"),
    pd.Series(predicted_day_night_mask.index, predicted_day_night_mask.index,
              name="datetime"), predicted_day_night_mask], axis=1)
sunrise = predicted_day_night_df[predicted_day_night_df['ac_power__752']]\
    .groupby(['date', 'ac_power__752'])['datetime'].min()
sunset = predicted_day_night_df[predicted_day_night_df['ac_power__752']]\
    .groupby(['date', 'ac_power__752'])['datetime'].max()
predicted_sunrise_sunset = pd.concat([sunrise.rename('predicted_sunrise'),
                                      sunset.rename('predicted_sunset')
                                      ], axis=1).reset_index(drop=True)
predicted_sunrise_sunset = predicted_sunrise_sunset.set_index(
    predicted_sunrise_sunset.predicted_sunrise.dt.date)
gt_sunrise_sunset = sunrise_sunset_df[['sunrise',
                                       'sunset']].\
    drop_duplicates().reset_index(drop=True)
gt_sunrise_sunset = gt_sunrise_sunset.set_index(
    gt_sunrise_sunset.sunrise.dt.date)

# Compare predicted sunrise and sunset times to ground truth sunrise
# and sunset times
sunrise_sunset_comparison_df = pd.concat([predicted_sunrise_sunset,
                                          gt_sunrise_sunset], axis=1)
print("Comparing sunrise times:")
print(sunrise_sunset_comparison_df[['sunrise', 'predicted_sunrise']])
print("Comparing sunset times:")
print(sunrise_sunset_comparison_df[['sunset', 'predicted_sunset']])
