"""
Infer System Orientation (Azimuth & Tilt) using Daily Peak
==========================================================

Infer the azimuth and tilt of a system using its daily peak value
"""

# %%
# Identifing and removing stale, or consecutive repeating, values in time
# series data reduces noise when performing data analysis. This example shows
# how to use two PVAnalytics functions,
# :py:func:`pvanalytics.quality.gaps.stale_values_diff`
# and :py:func:`pvanalytics.quality.gaps.stale_values_round`, to identify
# and mask stale data periods in time series data.

import pvanalytics
from pvanalytics.features.clipping import geometric
from pvanalytics.features.daytime import power_or_irradiance
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

# %%
# First, we import the AC power data stream that we are going to check for
# stale data periods. The time series we download is a normalized AC power time
# series from the PV Fleets Initiative, and is available via the DuraMAT
# DataHub:
# https://datahub.duramat.org/dataset/inverter-clipping-ml-training-set-real-data
# This data set has a Pandas DateTime index, with the min-max normalized
# AC power time series represented in the 'value_normalized' column.
# Additionally, there is a "stale_data_mask" column, where stale periods are
# labeled as True, and all other data is labeled as False. The data
# is sampled at 15-minute intervals.

pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
file = pvanalytics_dir / 'data' / 'ac_power_inv_2173_stale_data.csv'
data = pd.read_csv(file, index_col=0, parse_dates=True)
data = data.asfreq("15T")
data['value_normalized'].plot()
data.loc[data["stale_data_mask"], "value_normalized"].plot(ls='', marker='.')
plt.legend(labels=["AC Power", "Inserted Stale Data"])
plt.xlabel("Date")
plt.ylabel("Normalized AC Power")
plt.tight_layout()
plt.show()

# %%
# Run the clipping and the daytime filters on the time series. 
# Both of these masks will be used as inputs to the 
# :py:func:`pvanalytics.system.is_tracking_envelope` function. 

# Generate the daylight mask for the AC power time series
daytime_mask = day.power_or_irradiance(time_series)
# Get the sunny days associated with the system
sunny_days = ort.fixed_nrel(time_series,
               daytime_mask, r2_min=0.94,
               min_hours=5, peak_min=None)
# Run the PVAnalytics algo on it
tilts = [*range(0, 65, 5)]
azimuths = [*range(90,270, 5)]
# Get solar azimuth + zenith +ghi + dhi + dni from pvlib, based on 
# lat-long coords
sun = pvlib.solarposition.get_solarposition(time_series.index,
                                            latitude,
                                            longitude)
#get clear sky irradiance for the df index
loc = pvlib.location.Location(latitude,
                              longitude,
                              altitude=elevation)
CS = loc.get_clearsky(time_series.index)
# Run it through the orientation algorithm
best_azimuth, best_tilt = sys.infer_orientation_daily_peak(time_series, sunny_days, 
                                                           tilts,
                                                           azimuths, sun.azimuth,
                                                           sun.zenith, CS.ghi, CS.dhi, CS.dni)
print("Actual Azimuth: " + str(azimuth))
print("Predicted Azimuth: " + str(best_azimuth))
print("Actual Tilt: " + str(tilt))
print("Predicted Tilt: " + str(best_tilt))

