"""
Clearky Limits for Irradiance Data
==================================

Checking the clearsky limits of irradiance data.
"""

# %%
# Identifying and filtering out invalid irradiance data is a
# useful way to reduce noise during analysis. In this example,
# we use :py:func:`pvanalytics.quality.clearksy_limits`
# to identify irradiance values that do not exceed
#clearsky values. For this example we'll use
# GHI measurements from NREL in Golden CO.

import pvanalytics
from pvanalytics.quality.irradiance import clearsky_limits
import pvlib
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

# %%
# First, read in the RMIS NREL system. This data set contains
# 5-minute right-aligned sampled data. It includes POA, GHI,
# DNI, DHI, and GNI measurements.

pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
rmis_file = "C:/Users/kperry/Documents/source/repos/pvanalytics/pvanalytics/data/irradiance_RMIS_NREL.csv"#pvanalytics_dir / 'data' / 'irradiance_RMIS_NREL.csv'
data = pd.read_csv(rmis_file, index_col=0, parse_dates=True)

# %%
# Now model clear-sky irradiance for the location and times of the
# measured data:
location = pvlib.location.Location(39.7407, -105.1686)
clearsky = location.get_clearsky(data.index)

# %%
# Use :py:func:`pvanalytics.quality.irradiance.clearsky_limits`
# Here, we check GHI irradiance field 'irradiance_ghi__7981'.

clearsky_limit_mask = clearsky_limits(data['irradiance_ghi__7981'],
                                      clearsky['ghi'])