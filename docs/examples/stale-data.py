"""
Stale Data Periods
===================

Identifying stale data periods, defined as periods of
consecutive repeating values, in time series.
"""

# %%
# Identifing and removing stale, or consecutive repeating, values in time
# series data reduces noise when performing data analysis. This example shows
# how to use two PVAnalytics functions,
# :py:func:`pvanalytics.features.gaps.stale_values_diff`
# and :py:func:`pvanalytics.features.gaps.stale_values_round`, to identify
# and mask stale data periods in time series data.

import pvanalytics
from pvanalytics.quality import gaps
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import numpy as np

# %%
# First, we import the AC power data stream that we are going to check for
# stale data periods. The time series we download is a normalized AC power time
# series from the PV Fleets Initiative, and is available via the DuraMAT
# DataHub:
# https://datahub.duramat.org/dataset/inverter-clipping-ml-training-set-real-data

pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
file = pvanalytics_dir / 'data' / 'ac_power_inv_2173.csv'
data = pd.read_csv(file, index_col=0, parse_dates=True)
data = data.asfreq("15T")

# %%
# We plot the time series before inserting artificial stale data periods.
data.plot()
plt.xlabel("Date")
plt.ylabel("Normalized AC Power")
plt.legend(labels=["AC Power"])
plt.tight_layout()
plt.show()

# %%
# We insert some repeating/stale data periods into the time series for the
# stale data functions to catch, and re-visualize the data, with those stale
# periods masked

data[460:520] = data.iloc[460]
data[755:855] = data.iloc[755]
data[1515:1600] = data.iloc[1515]
stale_data_insert_mask = pd.Series([False] * len(data), index=data.index)
stale_data_insert_mask.iloc[np.r_[460:520, 755:855, 1515:1600]] = True

data['value_normalized'].plot()
data.loc[stale_data_insert_mask, "value_normalized"].plot(ls='', marker='.')
plt.legend(labels=["AC Power", "Inserted Stale Data"])
plt.xlabel("Date")
plt.ylabel("Normalized AC Power")
plt.tight_layout()
plt.show()

# %%
# Now, we use :py:func:`pvanalytics.features.gaps.stale_values_diff` to identify
# stale values in data. We visualize the detected stale periods graphicallyy.

stale_data_mask = gaps.stale_values_diff(data['value_normalized'])
data['value_normalized'].plot()
data.loc[stale_data_mask, "value_normalized"].plot(ls='', marker='.')
plt.legend(labels=["AC Power", "Detected Stale Data"])
plt.xlabel("Date")
plt.ylabel("Normalized AC Power")
plt.tight_layout()
plt.show()

# %%
# Now, we use :py:func:`pvanalytics.features.gaps.stale_values_round` to identify
# stale values in data, using rounded data. This function yields similar
# results as :py:func:`pvanalytics.features.gaps.stale_values_diff`, except it
# looks for consecutive repeating data that has been rounded to a settable
# decimals place.

stale_data_round_mask = gaps.stale_values_round(data['value_normalized'])
data['value_normalized'].plot()
data.loc[stale_data_round_mask, "value_normalized"].plot(ls='', marker='.')
plt.legend(labels=["AC Power", "Detected Stale Data"])
plt.xlabel("Date")
plt.ylabel("Normalized AC Power")
plt.tight_layout()
plt.show()
