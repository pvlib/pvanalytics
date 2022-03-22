"""
Day-Night Masking
=================

Masking day-night periods using the PVAnalytics daytime module.
"""

# %%
# Identifying and masking day-night periods in an AC power time series or
# irradiance time series can aid in future data analysis, such as detecting
# if a time series has daylight savings time or time shifts.

import pvanalytics
from pvanalytics.features.daytime import power_or_irradiance
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

# %%
# First, read in the 1-minute sampled AC power time series data, taken
# from the SERF East installation on the NREL campus.
# This sample is provided from the NREL PVDAQ database, and contains
# a column representing an AC power data stream, and a PVLib-derived day-night
# mask column. The pvlib.solarposition.sun_rise_set_transit_spa function was
# used to get sunrise and sunset times for each day at the site location, and
# the daytime mask was calculated based on these times. Data associated with
# day time periods is labeled as True, and data associated with nighttime
# periods is labeled as False.

pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
ac_power_file_1 = "C:/Users/kperry/Documents/source/repos/pvanalytics/pvanalytics/data/ac_power_inv_7539.csv"#pvanalytics_dir / 'data' / 'ac_power_inv_7539.csv'
data = pd.read_csv(ac_power_file_1, index_col=0, parse_dates=True)
data['label'] = data['label'].astype(bool)
# This is the known frequency of the time series. You may need to infer
# the frequency or set the frequency with your AC power time series.
freq = "1T"
data['value_normalized'].plot()
data.loc[data['label'], 'value_normalized'].plot(ls='', marker='o')
plt.legend(labels=["AC Power", "Daytime"])
plt.xticks(rotation=20)
plt.xlabel("Date")
plt.ylabel("Normalized AC Power")
plt.tight_layout()
plt.show()

# %%
# Now, use :py:func:`pvanalytics.features.daytime.power_or_irradiance`
# to identify day periods in the time series. Re-plot the data
# subset with this mask.
predicted_day_night_mask = power_or_irradiance(series=data['value_normalized'],
                                               freq=freq)
data['value_normalized'].plot()
data.loc[predicted_day_night_mask, 'value_normalized'].plot(ls='', marker='o')
data.loc[~predicted_day_night_mask, 'value_normalized'].plot(ls='', marker='o')
plt.legend(labels=["AC Power", "Daytime", "Nighttime"])
plt.xticks(rotation=20)
plt.xlabel("Date")
plt.ylabel("Normalized AC Power")
plt.tight_layout()
plt.show()


# %%
# Now read in the 15-minute sampled AC power time series data, taken
# from the SERF East installation on the NREL campus. This data is derived
# from the 1-minute data, except it has been resampled to 15-minute
# right-aligned mean data.
# Once again, there is a column representing an AC power data stream, and a
# PVLib-derived day-night mask column. The
# pvlib.solarposition.sun_rise_set_transit_spa function was
# used to get sunrise and sunset times for each day at the site location, and
# the daytime mask was calculated based on these times. Data associated with
# day time periods is labeled as True, and data associated with nighttime
# periods is labeled as False.
ac_power_file_2 = "C:/Users/kperry/Documents/source/repos/pvanalytics/pvanalytics/data/ac_power_inv_7539.csv"#pvanalytics_dir / 'data' / 'ac_power_inv_7539.csv'
data = pd.read_csv(ac_power_file_1, index_col=0, parse_dates=True)
data['label'] = data['label'].astype(bool)
# This is the known frequency of the time series. You may need to infer
# the frequency or set the frequency with your AC power time series.
freq = "15T"
data['value_normalized'].plot()
data.loc[data['label'], 'value_normalized'].plot(ls='', marker='o')
plt.legend(labels=["AC Power", "Daytime"])
plt.xticks(rotation=20)
plt.xlabel("Date")
plt.ylabel("Normalized AC Power")
plt.tight_layout()
plt.show()

# %%
# Now, use :py:func:`pvanalytics.features.daytime.power_or_irradiance`
# to identify day periods in the time series. Re-plot the data
# subset with this mask.
predicted_day_night_mask = power_or_irradiance(series=data['value_normalized'],
                                               freq=freq)
data['value_normalized'].plot()
data.loc[predicted_day_night_mask, 'value_normalized'].plot(ls='', marker='o')
data.loc[~predicted_day_night_mask, 'value_normalized'].plot(ls='', marker='o')
plt.legend(labels=["AC Power", "Daytime", "Nighttime"])
plt.xticks(rotation=20)
plt.xlabel("Date")
plt.ylabel("Normalized AC Power")
plt.tight_layout()
plt.show()
