"""
Clearsky Limits for Irradiance Data
===================================

Checking the clearsky limits of irradiance data.
"""

# %%
# Identifying and filtering out invalid irradiance data is a
# useful way to reduce noise during analysis. In this example,
# we use :py:func:`pvanalytics.quality.irradiance.clearsky_limits`
# to identify irradiance values that do not exceed
# a limit based on a clear-sky model. For this example we will use GHI data from the RMIS weather
# system located on the NREL campus in CO.

import pvanalytics
from pvanalytics.quality.irradiance import clearsky_limits
import pvlib
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

# %%
# First, read in the RMIS NREL system. This data set contains
# 5-minute right-aligned data. It includes POA, GHI,
# DNI, DHI, and GNI measurements, but only the GHI is relevant here.

pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
rmis_file = pvanalytics_dir / 'data' / 'irradiance_RMIS_NREL.csv'
data = pd.read_csv(rmis_file, index_col=0, parse_dates=True)
# Make the datetime index tz-aware.
data.index = data.index.tz_localize("Etc/GMT+7")


# %%
# Now model clear-sky irradiance for the location and times of the
# measured data. You can do this using
# :py:meth:`pvlib.location.Location.get_clearsky`, using the lat-long
# coordinates associated the RMIS NREL system.

location = pvlib.location.Location(39.7407, -105.1686)
clearsky = location.get_clearsky(data.index)

# %%
# Use :py:func:`pvanalytics.quality.irradiance.clearsky_limits`.
# Here, we check GHI field 'irradiance_ghi__7981'.

clearsky_limit_mask = clearsky_limits(data['irradiance_ghi__7981'],
                                      clearsky['ghi'])


# %%
# Plot the 'irradiance_ghi__7981' data stream and its associated clearsky GHI
# data stream. Mask the GHI time series by its clearsky_limit_mask.
data['irradiance_ghi__7981'].plot()
clearsky['ghi'].plot()
data.loc[clearsky_limit_mask]['irradiance_ghi__7981'].plot(ls='', marker='.')
plt.legend(labels=["RMIS GHI", "Clearsky GHI",
                   "Under Clearsky Limit"],
           loc="upper left")
plt.xlabel("Date")
plt.ylabel("GHI (W/m^2)")
plt.tight_layout()
plt.show()
