"""
QCrad Limits for Irradiance Data
================================

Test for physical limits on GHI, DHI or DNI using the QCRad criteria.
"""

# %%
# Identifying and filtering out invalid irradiance data is a
# useful way to reduce noise during analysis. In this example,
# we use
# :py:func:`pvanalytics.quality.irradiance.check_irradiance_limits_qcrad`
# to test for physical limits on GHI, DHI or DNI using the QCRad criteria.
# For this example we will use data from the RMIS weather system located
# on the NREL campus in Colorado, USA.

import pvanalytics
from pvanalytics.quality.irradiance import check_irradiance_limits_qcrad
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
# coordinates. This is done using the
# :py:func:`pvlib.solarposition.get_solarposition` function.
latitude = 39.742
longitude = -105.18
time_zone = "Etc/GMT+7"
data = data.tz_localize(time_zone)
solar_position = pvlib.solarposition.get_solarposition(data.index,
                                                       latitude,
                                                       longitude)

# %%
# Generate the estimated extraterrestrial radiation for the time series,
# referred to as dni_extra. This is done using the
# :py:func:`pvlib.irradiance.get_extra_radiation` function.
dni_extra = pvlib.irradiance.get_extra_radiation(data.index)

# %%
# Use :py:func:`pvanalytics.quality.irradiance.check_irradiance_limits_qcrad`
# to generate the QCRAD irradiance limit mask

qcrad_limit_mask = check_irradiance_limits_qcrad(
    solar_zenith=solar_position['zenith'],
    dni_extra=dni_extra,
    ghi=data['irradiance_ghi__7981'],
    dhi=data['irradiance_dhi__7983'],
    dni=data['irradiance_dni__7982'])

# %%
# Plot the 'irradiance_ghi__7981' data stream with its associated QCRAD limit
# mask.
data['irradiance_ghi__7981'].plot()
data.loc[qcrad_limit_mask[0], 'irradiance_ghi__7981'].plot(ls='', marker='.')
plt.legend(labels=["RMIS GHI", "Within QCRAD Limits"],
           loc="upper left")
plt.xlabel("Date")
plt.ylabel("GHI (W/m^2)")
plt.tight_layout()
plt.show()

# %%
# Plot the 'irradiance_dhi__7983 data stream with its associated QCRAD limit
# mask.
data['irradiance_dhi__7983'].plot()
data.loc[qcrad_limit_mask[1], 'irradiance_dhi__7983'].plot(ls='', marker='.')
plt.legend(labels=["RMIS DHI", "Within QCRAD Limits"],
           loc="upper left")
plt.xlabel("Date")
plt.ylabel("DHI (W/m^2)")
plt.tight_layout()
plt.show()

# %%
# Plot the 'irradiance_dni__7982' data stream with its associated QCRAD limit
# mask.
data['irradiance_dni__7982'].plot()
data.loc[qcrad_limit_mask[2], 'irradiance_dni__7982'].plot(ls='', marker='.')
plt.legend(labels=["RMIS DNI", "Within QCRAD Limits"],
           loc="upper left")
plt.xlabel("Date")
plt.ylabel("DNI (W/m^2)")
plt.tight_layout()
plt.show()
