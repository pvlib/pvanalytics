"""
Module Temperature Check
========================

Test whether the module temperature is correlated with irradiance.
"""

# %%
# Testing the correlation between module temperature and irradiance
# measurements can help identify if there are issues with the module
# temperature sensor.
# In this example, we demonstrate how to use
# :py:func:`pvanalytics.quality.weather.module_temperature_check`, which
# runs a linear regression model for module temperature vs irradiance. The
# model  is then assessed by correlation coefficient. If it meets a minimum
# threshold, function outputs a True boolean. If not, it outputs a False
# boolean.

import pvanalytics
from pvanalytics.quality.weather import module_temperature_check
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
import pathlib

# %%
# First, we read in the NREL SERF West system example, which contains
# data for module temperature and irradiance under the 'module_temp_1__781'
# and 'poa_irradiance__771' columns, respectively. This data set contains
# 1-minute measurements.
pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
serf_east_file = pvanalytics_dir / 'data' / 'serf_west_1min.csv'
data = pd.read_csv(serf_east_file, index_col=0, parse_dates=True)
print(data[['module_temp_1__781', 'poa_irradiance__771']].head(10))

# %%
# We then use :py:func:`pvanalytics.quality.weather.module_temperature_check`
# to regress module temperature against irradiance POA, and check if the
# relationships meets the minimum correlation coefficient criteria.
corr_coeff_bool = module_temperature_check(data['module_temp_1__781'],
                                           data['poa_irradiance__771'])
print("Passes R^2 threshold? " + str(corr_coeff_bool))

# %%
# Plot module temperature against irradiance to illustrate the relationship
data.plot(x='module_temp_1__781',
          y='poa_irradiance__771',
          style='o', legend=None)
data_reg = data[['module_temp_1__781', 'poa_irradiance__771']].dropna()
reg = linregress(data_reg['module_temp_1__781'].values,
                 data_reg['poa_irradiance__771'].values)
plt.axline(xy1=(0, reg.intercept), slope=reg.slope, linestyle="--", color="k")
# Add the linear regression line with R^2
plt.xlabel("Module Temperature (deg C)")
plt.ylabel("POA irradiance (W/m^2)")
plt.tight_layout()
plt.show()
