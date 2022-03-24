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
# Now generate solar zenith measurements for the location,
# based on the data's time zone and site latitude-longitude
# coordinates.
latitude = 39.742
longitude = -105.18
time_zone = "Etc/GMT+7"
CSi = data.index.tz_localize(time_zone,
                             ambiguous='NaT',
                             nonexistent='NaT')
solar_position = pvlib.solarposition.get_solarposition(CSi,
                                                       latitude,
                                                       longitude)

# %%
# Use :py:func:`pvanalytics.quality.irradiance.daily_insolation_limits`
# to identify if the daily insolation lies between a minimum
# and a maximum value. Here, we check POA irradiance field
# 'irradiance_poa__7984'.

check_irradiance_consistency_qcrad(ghi=data['irradiance_ghi__7981'],
                                   solar_zenith=solar_position['zenith'],
                                   dhi=data['irradiance_dhi__7983'],
                                   dni=data['irradiance_dni__7982'])
