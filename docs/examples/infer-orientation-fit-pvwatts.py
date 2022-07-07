"""
Infer System Orientation (Azimuth & Tilt) using PVWatts
=======================================================

Infer the azimuth and tilt of a system using PVWatts data
"""

# %%
# Identifing and/or validating  the azimuth and tilt information for a
# system is important, as these values must be correct for future degradation
# and system yield analysis. This example shows how to use
# :py:func:`pvanalytics.system.infer_orientation_daily_peak` to estimate
# a system's azimuth and tilt, using the system's known latitude-longitude
# coordinates and an associated AC power time series.

from pvanalytics.features.daytime import power_or_irradiance
from pvanalytics.features.orientation import fixed_nrel
from pvanalytics import system as sys
import pandas as pd
import pathlib
import pvlib

# %%
# First, we import the AC power data stream from the SERF East site located at 
# NREL. This data set is publicly available via the PVDAQ database in the
# DOE Open Energy Data Initiative (OEDI)
# (https://data.openei.org/submissions/4568).

dataframe = pd.read_csv("C:/Users/kperry/Documents/Wells Fargo/code/50.csv")
dataframe.index = pd.to_datetime(dataframe['measured_on'])
time_series = dataframe['ac_power__752']
time_series = time_series.asfreq('T')

# Outline the ground truth metadata associated with the system
latitude = 39.742
longitude = -105.1727
elevation =0
actual_azimuth = 158
actual_tilt = 45

# %%
# Run the clipping and the daytime filters on the time series. 
# Both of these masks will be used as inputs to the 
# :py:func:`pvanalytics.system.infer_orientation_daily_peak` function. 

# Generate the daylight mask for the AC power time series
daytime_mask = power_or_irradiance(time_series)

# Get the sunny days associated with the system
sunny_days = fixed_nrel(time_series,
                        daytime_mask)

# Filter the time series to only include data on clearsky days
time_series_clearsky = time_series[sunny_days]
time_series_clearsky = time_series_clearsky.dropna()

# Get solar azimuth + zenith +ghi + dhi + dni from pvlib, based on 
# lat-long coords
sun = pvlib.solarposition.get_solarposition(time_series_clearsky.index,
                                            latitude,
                                            longitude)

#Get clear sky irradiance for the time series index
loc = pvlib.location.Location(latitude,
                              longitude)
CS = loc.get_clearsky(time_series_clearsky.index)

# Run the pvlib data and the sensor-based time series data through 
# the :py:func:`pvanalytics.system.infer_orientation_daily_peak` function.
best_tilt, best_azimuth, r2 = sys.infer_orientation_fit_pvwatts(time_series_clearsky,
                                                      CS.ghi,
                                                      CS.dhi,
                                                      CS.dni,
                                                      sun.zenith,
                                                      sun.azimuth)
print("Actual Azimuth: " + str(actual_azimuth))
print("Predicted Azimuth: " + str(best_azimuth))
print("Actual Tilt: " + str(actual_tilt))
print("Predicted Tilt: " + str(best_tilt))

