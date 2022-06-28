"""
Infer System Orientation (Azimuth & Tilt) using PVWatts
=======================================================

Infer the azimuth and tilt of a system using PVWatts data
"""

# %%
# Identifing and/or validating the azimuth and tilt details for 
# a system is critical for analysis. This example uses the 

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
daytime_mask = power_or_irradiance(time_series)
# Get the sunny days associated with the system
sunny_days = fixed_nrel(time_series,
                        daytime_mask, r2_min=0.94,
                        min_hours=5, peak_min=None)
# Run the PV Analytics algo on it
tilts = [*range(0, 65, 5)]
azimuths = [*range(90,270, 5)]



# %%
# Now, we use :py:func:`pvanalytics.quality.gaps.stale_values_round` to
# identify stale values in data, using rounded data. This function yields
# similar results as :py:func:`pvanalytics.quality.gaps.stale_values_diff`,
# except it looks for consecutive repeating data that has been rounded to
# a settable decimals place.
# Please note that nighttime periods generally
# contain consecutive repeating 0 values, which are flagged by
# :py:func:`pvanalytics.quality.gaps.stale_values_round`.

stale_data_round_mask = gaps.stale_values_round(data['value_normalized'])
data['value_normalized'].plot()
data.loc[stale_data_round_mask, "value_normalized"].plot(ls='', marker='.')
plt.legend(labels=["AC Power", "Detected Stale Data"])
plt.xlabel("Date")
plt.ylabel("Normalized AC Power")
plt.tight_layout()
plt.show()
