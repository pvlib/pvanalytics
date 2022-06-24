"""
Z-Score Outlier Detection
=========================

Identifying outliers in time series using
z-score outlier detection.
"""

# %%
# Identifying and removing outliers from PV sensor time series
# data allows for more accurate data analysis.
# In this example, we demonstrate how to use
# :py:func:`pvanalytics.quality.outliers.zscore` to identify and filter
# out outliers in a time series.

import pvanalytics
from pvanalytics.quality.outliers import zscore
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

# %%
# First, we read in the ac_power_inv_7539_outliers example. Min-max normalized
# AC power is represented by the "value_normalized" column. There is a boolean
# column "outlier" where inserted outliers are labeled as True, and all other
# values are labeled as False. These outlier values were inserted manually into
# the data set to illustrate outlier detection by each of the functions.
# We use a normalized time series example provided by the PV Fleets Initiative.
# This example is adapted from the DuraMAT DataHub
# clipping data set:
# https://datahub.duramat.org/dataset/inverter-clipping-ml-training-set-real-data
pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
ac_power_file = pvanalytics_dir / 'data' / 'ac_power_inv_7539_outliers.csv'
data = pd.read_csv(ac_power_file, index_col=0, parse_dates=True)
print(data.head(10))

# %%
# We then use :py:func:`pvanalytics.quality.outliers.zscore` to identify
# outliers in the time series, and plot the data with the z-score outlier mask.
zscore_outlier_mask = zscore(data=data['value_normalized'])
data['value_normalized'].plot()
data.loc[zscore_outlier_mask, 'value_normalized'].plot(ls='', marker='o')
plt.legend(labels=["AC Power", "Detected Outlier"])
plt.xlabel("Date")
plt.ylabel("Normalized AC Power")
plt.tight_layout()
plt.show()
