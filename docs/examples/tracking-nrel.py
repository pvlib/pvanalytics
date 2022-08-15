"""
Flag Sunny Days for a Tracking System
=====================================

Flag sunny days for a single-axis tracking PV system.
"""

# %%
# Identifying and masking sunny days for a single-axis tracking system is
# important when performing future analyses that require filtered sunny day
# data. For this example we will use data from the single-axis tracking
# NREL Mesa system located on the NREL campus in Colorado, USA, and generate
# a sunny day mask.
# This data set is publicly available via the PVDAQ database in the
# DOE Open Energy Data Initiative (OEDI)
# (https://data.openei.org/submissions/4568), as system ID 50.
# This data is timezone-localized.

import pvanalytics
from pvanalytics.features import daytime as day
from pvanalytics.features.orientation import tracking_nrel
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

# %%
# First, read in data from the NREL Mesa 1-axis tracking system. This data
# set contains 15-minute interval AC power data.

pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
file = pvanalytics_dir / 'data' / 'nrel_1axis_tracker_mesa_ac_power.csv'
data = pd.read_csv(file, index_col=0, parse_dates=True)

# %%
# Mask day-night periods using the
# :py:func:`pvanalytics.features.daytime.power_or_irradiance` function.
# Then apply :py:func:`pvanalytics.features.orientation.tracking_nrel`
# to the AC power stream and mask the sunny days in the time series.

daytime_mask = day.power_or_irradiance(data['ac_power'])

tracking_sunny_days = tracking_nrel(data['ac_power'],
                                    daytime_mask)

# %%
# Plot the AC power stream with the sunny day mask applied to it.

data['ac_power'].plot()
data.loc[tracking_sunny_days, 'ac_power'].plot(ls='', marker='.')
plt.legend(labels=["AC Power", "Sunny Day"],
           loc="upper left")
plt.xlabel("Date")
plt.ylabel("AC Power (kW)")
plt.tight_layout()
plt.show()
