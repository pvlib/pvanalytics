"""
Wind Limits
===========

Identifying wind speed values that are within limits.
"""

# %%
# Identifying wind speed values that are within logical, expected limits
# and filtering data outside of these limits allows for more accurate
# future data analysis.
# In this example, we demonstrate how to use
# :py:func:`pvanalytics.quality.weather.wind_limits` to identify and
# filter out wind speed values that are not within expected limits.

import pvanalytics
from pvanalytics.quality.weather import wind_limits
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

# %%
# First, we read in the NREL RMIS weather station example, which contains
# wind speed data in m/s under the column
# 'Wind Speed'. This data set contains 5-minute right-aligned
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
wind_limit_mask = wind_limits(data['Wind Speed'])
data['Wind Speed'].plot()
data.loc[~wind_limit_mask, 'Wind Speed'].plot(ls='', marker='o')
plt.legend(labels=["Wind Speed", "Detected Outlier"])
plt.xlabel("Date")
plt.ylabel("Wind Speed (m/s)")
plt.tight_layout()
plt.show()
