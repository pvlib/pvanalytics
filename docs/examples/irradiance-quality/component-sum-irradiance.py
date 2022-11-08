"""
Component Sum Equations for Irradiance Data
===========================================

Estimate GHI, DHI, and DNI using the component sum equations, with
nighttime corrections.
"""

# %%
# Estimating GHI, DHI, and DNI using the component sum equations is useful
# if the associated field is missing, or as a comparison to an existing
# physical data stream.

import pvanalytics
from pvanalytics.quality.irradiance import calculate_component_sum_series
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
# Get the clearsky DNI values associated with the current location, using
# the :py:func:`pvlib.solarposition.get_solarposition` function. These clearsky
# values are used to calculate DNI data.
site = pvlib.location.Location(latitude, longitude, tz=time_zone)
clearsky = site.get_clearsky(data.index)

# %%
# Use :py:func:`pvanalytics.quality.irradiance.calcuate_ghi_component`
# to estimate GHI measurements using DHI and DNI measurements

component_sum_ghi = calculate_component_sum_series(
    solar_zenith=solar_position['zenith'],
    dhi=data['irradiance_dhi__7983'],
    dni=data['irradiance_dni__7982'],
    zenith_limit=90,
    fill_night_value='equation')

# %%
# Plot the 'irradiance_ghi__7981' data stream against the estimated component
# sum GHI, for comparison
data['irradiance_ghi__7981'].plot()
component_sum_ghi.plot()
plt.legend(labels=["RMIS GHI", "Component Sum GHI"],
           loc="upper left")
plt.xlabel("Date")
plt.ylabel("GHI (W/m^2)")
plt.tight_layout()
plt.show()

# %%
# Use :py:func:`pvanalytics.quality.irradiance.calcuate_dhi_component`
# to estimate DHI measurements using GHI and DNI measurements

component_sum_dhi = calculate_component_sum_series(
    solar_zenith=solar_position['zenith'],
    dni=data['irradiance_dni__7982'],
    ghi=data['irradiance_ghi__7981'],
    zenith_limit=90,
    fill_night_value='equation')

# %%
# Plot the 'irradiance_dhi__7983' data stream against the estimated component
# sum GHI, for comparison
data['irradiance_dhi__7983'].plot()
component_sum_dhi.plot()
plt.legend(labels=["RMIS DHI", "Component Sum DHI"],
           loc="upper left")
plt.xlabel("Date")
plt.ylabel("GHI (W/m^2)")
plt.tight_layout()
plt.show()

# %%
# Use :py:func:`pvanalytics.quality.irradiance.calcuate_dni_component`
# to estimate DNI measurements using GHI and DHI measurements

component_sum_dni = calculate_component_sum_series(
    solar_zenith=solar_position['zenith'],
    dhi=data['irradiance_dhi__7983'],
    ghi=data['irradiance_ghi__7981'],
    dni_clear=clearsky['dni'],
    zenith_limit=90,
    fill_night_value='equation')

# %%
# Plot the 'irradiance_dni__7982' data stream against the estimated component
# sum GHI, for comparison
data['irradiance_dni__7982'].plot()
component_sum_dni.plot()
plt.legend(labels=["RMIS DNI", "Component Sum DNI"],
           loc="upper left")
plt.xlabel("Date")
plt.ylabel("DNI (W/m^2)")
plt.tight_layout()
plt.show()
