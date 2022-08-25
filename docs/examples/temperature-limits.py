"""
Air Temperature Limits
======================

Identifying air/ambient temperature values that are within limits.
"""

# %%
# Identifying air temperature values that are within logical, expected limits
# and filtering data outside of these limits allows for more accurate
# future data analysis.
# In this example, we demonstrate how to use
# :py:func:`pvanalytics.quality.weather.temperature_limits` to identify and
# filter out air temperature values that are not within expected limits.

import pvanalytics
from pvanalytics.quality.weather import temperature_limits
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

# %%
# First, we read in the NREL RMIS weather station example, which contains
# air temperature data in degrees Celsius under the column
# 'Ambient Temperature'. This data set contains 5-minute right-aligned
# measurements.
pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
rmis_file = pvanalytics_dir / 'data' / 'rmis_weather_data.csv'
data = pd.read_csv(rmis_file, index_col=0, parse_dates=True)
print(data.head(10))

# %%
# We then use :py:func:`pvanalytics.quality.weather.temperature_limits`
# to identify any air temperature values that are not within an
# acceptable range. We can then filter any of these values out of the time
# series.
temperature_limit_mask = temperature_limits(data['Ambient Temperature'])
data['Ambient Temperature'].plot()
data.loc[~temperature_limit_mask, 'Ambient Temperature'].plot(ls='',
                                                              marker='o')
plt.legend(labels=["Ambient Temperature", "Detected Outlier"])
plt.xlabel("Date")
plt.ylabel("Ambient Temperature (deg C)")
plt.tight_layout()
plt.show()
