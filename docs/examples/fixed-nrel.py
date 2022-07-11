"""
Flag Sunny Days for a Fixed-Tilt System
=======================================

Flag sunny days for a fixed-tilt PV system.
"""

# %%
# Identifying and masking sunny days for a fixed-tilt system is important
# when performing future analyses that require filtered clearsky data.
# For this example we will use data from the fixed-tilt NREL SERF East system 
# located on the NREL campus in Colorado, USA, and generate a sunny day mask.

import pvanalytics
from pvanalytics.features import daytime as day
from pvanalytics.features.orientation import fixed_nrel
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

# %%
# First, read in data from the RMIS NREL system. This data set contains
# 5-minute right-aligned data. It includes POA, GHI,
# DNI, DHI, and GNI measurements.

pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
rmis_file = pvanalytics_dir / 'data' / 'irradiance_RMIS_NREL.csv'
data = pd.read_csv(rmis_file, index_col=0, parse_dates=True)
# Make the datetime index tz-aware.
data.index = data.index.tz_localize("Etc/GMT+7")

# %%
# First, mask day-night periods using xxx. Then apply XXX
# to the AC power stream and mask the sunny days in the time series.
daytime_mask = day.power_or_irradiance(data['ac_power'])

fixed_sunny_days = fixed_nrel(data['ac_power'],
                              daytime_mask)

# %%
# Plot a subset pf AC power stream with the sunny day mask applied to it.
data['ac_power'].plot()
data.loc[fixed_sunny_days, 'ac_power'].plot(ls='', marker='.')
plt.legend(labels=["AC Power", "Sunny Day"],
           loc="upper left")
plt.xlabel("Date")
plt.ylabel("AC Power (kW)")
plt.tight_layout()
plt.show()
