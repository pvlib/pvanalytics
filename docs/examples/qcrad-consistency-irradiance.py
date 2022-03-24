"""
QCrad Consistency for Irradiance Data
=====================================

Check consistency of GHI, DHI and DNI using QCRad criteria.
"""

# %%
# Identifying and filtering out invalid irradiance data is a
# useful way to reduce noise during analysis. In this example,
# we use :py:func:`pvanalytics.quality.check_irradiance_consistency_qcrad`
# to check the consistency of GHI, DHI and DNI data using QCRad criteria.

import pvanalytics
from pvanalytics.quality.irradiance import check_irradiance_consistency_qcrad
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
location = pvlib.location.Location(39.742, -105.18)
clearsky = location.get_clearsky(data.index)

# %%
# Use :py:func:`pvanalytics.quality.irradiance.daily_insolation_limits`
# to identify if the daily insolation lies between a minimum
# and a maximum value. Here, we check POA irradiance field
# 'irradiance_poa__7984'.

daily_insolation_limits(data['irradiance_poa__7984'],
                        clearsky)
