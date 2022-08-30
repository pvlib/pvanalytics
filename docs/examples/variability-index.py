"""
Calculate Variability Index
===========================

Calculate the Variability Index for a GHI time series.
"""

# %%
# The variability index provides a measure of variability for comparing
# solar sites and defining temporal patterns. This example uses GHI
# data collected from the NREL RMIS system to calculate the variability
# index as a time series.

import pvanalytics
from pvanalytics.metrics import variability_index
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import pvlib

# %%
# First, read in data from the RMIS NREL system. This data set contains
# 5-minute right-aligned POA, GHI, DNI, DHI, and GNI measurements,
# but only the GHI is relevant here.

pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
rmis_file = pvanalytics_dir / 'data' / 'irradiance_RMIS_NREL.csv'
data = pd.read_csv(rmis_file, index_col=0, parse_dates=True)
freq = '5T'
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
# Calculate the variability index for the system GHI data stream using
# the :py:func:`pvanalytics.metrics.variability_index` function.
variability_index_series = variability_index(data['irradiance_ghi__7981'],
                                             clearsky['ghi'],
                                             freq='5T')

# %%
# Plot the 5-minute interval variability index for the RMIS system.
variability_index_series.plot()
plt.xlabel("Date")
plt.ylabel("Variability Index")
plt.tight_layout()
plt.show()
