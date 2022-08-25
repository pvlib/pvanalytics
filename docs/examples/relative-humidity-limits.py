"""
Relative Humidity Limits
========================

Identifying relative humidity values that are within limits.
"""

# %%
# Identifying relative humidity (RH) values that are within logical,
# expected limits and filtering data outside of these limits allows for
# more accurate future data analysis.
# In this example, we demonstrate how to use
# :py:func:`pvanalytics.quality.weather.relative_humidity_limits` to
# identify and filter out RH values that are not within
# expected limits.

import pvanalytics
from pvanalytics.quality.weather import relative_humidity_limits
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

# %%
# First, we read in the NREL RMIS weather station example, which contains
# RH data in % under the column
# 'Relative Humidity'. This data set contains 5-minute right-aligned
# measurements.
pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
rmis_file = pvanalytics_dir / 'data' / 'rmis_weather_data.csv'
data = pd.read_csv(rmis_file, index_col=0, parse_dates=True)
print(data.head(10))

# %%
# We then use :py:func:`pvanalytics.quality.weather.relative_humidity_limits`
# to identify any RH values that are not within an
# acceptable range. We can then filter any of these values out of the time
# series.
rh_limit_mask = relative_humidity_limits(data['Relative Humidity'])
data['Relative Humidity'].plot()
data.loc[~rh_limit_mask, 'Relative Humidity'].plot(ls='', marker='o')
plt.legend(labels=['Relative Humidity', "Detected Outlier"])
plt.xlabel("Date")
plt.ylabel('Relative Humidity (%)')
plt.tight_layout()
plt.show()
