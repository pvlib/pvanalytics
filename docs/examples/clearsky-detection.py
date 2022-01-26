"""
Clear-Sky Detection
===================

Identifying periods of clear-sky conditions using measured irradiance.
"""

# %%
# Identifying and filtering for clear-sky conditions is a useful way to
# reduce noise when analyzing measured data.  This example shows how to
# use :py:func:`pvanalytics.features.clearsky.reno` to identify clear-sky
# conditions using measured GHI data.  For this example we'll retrieve
# GHI measurements from NREL in Golden CO.

from pvanalytics.features.clearsky import reno
from pvlib.iotools import read_midc_raw_data_from_nrel
import pvlib
import matplotlib.pyplot as plt
import pandas as pd

# %%
# First, fetch some example GHI data using pvlib.  For this example we'll just
# use a single day, but the same process applies to data of any length.

date = pd.to_datetime('2022-01-20')
data = read_midc_raw_data_from_nrel('BMS', start=date, end=date)
measured_ghi = data['Global CMP22 (vent/cor) [W/m^2]']

# %%
# Now model clear-sky irradiance for the location and times of the
# measured data:

location = pvlib.location.Location(39.742, -105.18)
clearsky = location.get_clearsky(data.index)
clearsky_ghi = clearsky['ghi']

# %%
# Finally, use :py:func:`pvanalytics.features.clearsky.reno` to identify
# measurements during clear-sky conditions:

is_clearsky = reno(measured_ghi, clearsky_ghi)

# clear-sky times indicated in black
measured_ghi.plot()
measured_ghi[is_clearsky].plot(ls='', marker='o', ms=2, c='k')
plt.ylabel('Global Horizontal Irradiance [W/m2]')
plt.show()
