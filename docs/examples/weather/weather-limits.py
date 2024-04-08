"""
Weather Limits
==============

Identifying weather values that are within limits.
"""

# %%
# Identifying weather values that are within logical, expected limits
# and filtering data outside of these limits allows for more accurate
# future data analysis.
# In this example, we demonstrate how to use
# :py:func:`pvanalytics.quality.weather.wind_limits`,
# :py:func:`pvanalytics.quality.weather.temperature_limits`,
# and :py:func:`pvanalytics.quality.weather.relative_humidity_limits`
# to identify and filter out values that are not within expected limits,
# for wind speed, ambient temperature, and relative humidity, respectively.

import pvanalytics
from pvanalytics.quality.weather import wind_limits, \
    temperature_limits, relative_humidity_limits
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

# %%
# First, we read in the NREL RMIS weather station example, which contains
# wind speed, temperature, and relative humidity data in m/s, deg C, and %
# respectively. This data set contains 5-minute right-aligned
# measurements.
pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
rmis_file = pvanalytics_dir / 'data' / 'rmis_weather_data.csv'
data = pd.read_csv(rmis_file, index_col=0, parse_dates=True)
print(data.head(10))

# %%
# First, we use :py:func:`pvanalytics.quality.weather.wind_limits`
# to identify any wind speed values that are not within an
# acceptable range. We can then filter any of these values out of the
# time series.
wind_limit_mask = wind_limits(data['Wind Speed'])
data['Wind Speed'].plot()
data.loc[~wind_limit_mask, 'Wind Speed'].plot(ls='', marker='o')
plt.legend(labels=["Wind Speed", "Detected Outlier"])
plt.xlabel("Date")
plt.ylabel("Wind Speed (m/s)")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()

# %%
# Next, we use :py:func:`pvanalytics.quality.weather.temperature_limits`
# to identify any air temperature values that are not within an
# acceptable range. We can then filter any of these values out of the time
# series. Here, we set the temperature limits to (-10,10), illustrating how
# to use the limits parameter.
temperature_limit_mask = temperature_limits(data['Ambient Temperature'],
                                            limits=(-10, 10))
data['Ambient Temperature'].plot()
data.loc[~temperature_limit_mask, 'Ambient Temperature'].plot(ls='',
                                                              marker='o')
plt.legend(labels=["Ambient Temperature", "Detected Outlier"])
plt.xlabel("Date")
plt.ylabel("Ambient Temperature (deg C)")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()

# %%
# Finally, we use
# :py:func:`pvanalytics.quality.weather.relative_humidity_limits`
# to identify any RH values that are not within an
# acceptable range. We can then filter any of these values out of the time
# series.
rh_limit_mask = relative_humidity_limits(data['Relative Humidity'])
data['Relative Humidity'].plot()
data.loc[~rh_limit_mask, 'Relative Humidity'].plot(ls='', marker='o')
plt.legend(labels=['Relative Humidity', "Detected Outlier"])
plt.xlabel("Date")
plt.ylabel('Relative Humidity (%)')
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()
