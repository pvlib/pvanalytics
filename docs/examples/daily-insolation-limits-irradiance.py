"""
Clearsky Limits for Daily Insolation
====================================

Checking the clearsky limits for daily insolation data.
"""

# %%
# Identifying and filtering out invalid irradiance data is a
# useful way to reduce noise during analysis. In this example,
# we use :py:func:`pvanalytics.quality.irradiance.daily_insolation_limits`
# to determine when the daily insolation lies between a minimum
# and a maximum value. Irradiance measurements and clear-sky
# irradiance on each day are integrated with the trapezoid rule
# to calculate daily insolation. For this example we will use data
# from the RMIS weather system located on the NREL campus
# in Colorado, USA.

import pvanalytics
from pvanalytics.quality.irradiance import daily_insolation_limits
import pvlib
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
# Now model clear-sky irradiance for the location and times of the
# measured data:
location = pvlib.location.Location(39.7407, -105.1686)
clearsky = location.get_clearsky(data.index)

# %%
# Use :py:func:`pvanalytics.quality.irradiance.daily_insolation_limits`
# to identify if the daily insolation lies between a minimum
# and a maximum value. Here, we check GHI irradiance field
# 'irradiance_ghi__7981'.
# :py:func:`pvanalytics.quality.irradiance.daily_insolation_limits`
# returns a mask that identifies data that falls between
# lower and upper limits. The defaults (used here)
# are upper bound of 125% of clear-sky daily insolation,
# and lower bound of 40% of clear-sky daily insolation.

daily_insolation_mask = daily_insolation_limits(data['irradiance_ghi__7981'],
                                                clearsky['ghi'])

# %%
# Plot the 'irradiance_ghi__7981' data stream and its associated clearsky GHI
# data stream. Mask the GHI time series by its daily_insolation_mask.
data['irradiance_ghi__7981'].plot()
clearsky['ghi'].plot()
data.loc[daily_insolation_mask, 'irradiance_ghi__7981'].plot(ls='', marker='.')
plt.legend(labels=["RMIS GHI", "Clearsky GHI",
                   "Within Daily Insolation Limit"],
           loc="upper left")
plt.xlabel("Date")
plt.ylabel("GHI (W/m^2)")
plt.tight_layout()
plt.show()
