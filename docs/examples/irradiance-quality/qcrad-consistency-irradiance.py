"""
QCrad Consistency for Irradiance Data
=====================================

Check consistency of GHI, DHI and DNI using QCRad criteria.
"""

# %%
# Identifying and filtering out invalid irradiance data is a
# useful way to reduce noise during analysis. In this example,
# we use
# :py:func:`pvanalytics.quality.irradiance.check_irradiance_consistency_qcrad`
# to check the consistency of GHI, DHI and DNI data using QCRad criteria.
# For this example we will use data from the RMIS weather system located
# on the NREL campus in Colorado, USA.


import pvanalytics
from pvanalytics.quality.irradiance import check_irradiance_consistency_qcrad
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

# %%
# Now generate solar zenith estimates for the location,
# based on the data's time zone and site latitude-longitude
# coordinates.
latitude = 39.742
longitude = -105.18
time_zone = "Etc/GMT+7"
data = data.tz_localize(time_zone)
solar_position = pvlib.solarposition.get_solarposition(data.index,
                                                       latitude,
                                                       longitude)

# %%
# Use
# :py:func:`pvanalytics.quality.irradiance.check_irradiance_consistency_qcrad`
# to generate the QCRAD consistency mask.

qcrad_consistency_mask = check_irradiance_consistency_qcrad(
    solar_zenith=solar_position['zenith'],
    ghi=data['irradiance_ghi__7981'],
    dhi=data['irradiance_dhi__7983'],
    dni=data['irradiance_dni__7982'])


# %%
# Plot the GHI, DHI, and DNI data streams with the QCRAD
# consistency mask overlay. This mask applies to all 3 data streams.
fig = data[['irradiance_ghi__7981', 'irradiance_dhi__7983',
            'irradiance_dni__7982']].plot()
# Highlight periods where the QCRAD consistency mask is True
fig.fill_between(data.index, fig.get_ylim()[0], fig.get_ylim()[1],
                 where=qcrad_consistency_mask[0], alpha=0.4)
fig.legend(labels=["RMIS GHI", "RMIS DHI", "RMIS DNI", "QCRAD Consistent"],
           loc="upper center")
plt.xlabel("Date")
plt.ylabel("Irradiance (W/m^2)")
plt.tight_layout()
plt.show()

# %%
# Plot the GHI, DHI, and DNI data streams with the diffuse
# ratio limit mask overlay. This mask is true when the
# DHI / GHI ratio passes the limit test.
fig = data[['irradiance_ghi__7981', 'irradiance_dhi__7983',
            'irradiance_dni__7982']].plot()
# Highlight periods where the GHI ratio passes the limit test
fig.fill_between(data.index, fig.get_ylim()[0], fig.get_ylim()[1],
                 where=qcrad_consistency_mask[1], alpha=0.4)
fig.legend(labels=["RMIS GHI", "RMIS DHI", "RMIS DNI",
                   "Within Diffuse Ratio Limit"],
           loc="upper center")
plt.xlabel("Date")
plt.ylabel("Irradiance (W/m^2)")
plt.tight_layout()
plt.show()
