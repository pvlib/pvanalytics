"""
Z-Score Outlier Detection
=========================

Identifying outliers in time series using
:py:func:`pvanalytics.quality.outliers.zscore`
"""

# %%
# Identifying and removing outliers from PV sensor time series
# data allows for more accurate data analysis.
# In this example, we demonstrate how to use
# :py:func:`pvanalytics.quality.outliers.tukey` to identify and filter
# out outliers in a time series.
# We use a normalized time series example provided by the PV Fleets Initiative.
# This example is adapted from the DuraMAT DataHub
# clipping data set:
# https://datahub.duramat.org/dataset/inverter-clipping-ml-training-set-real-data

import pvanalytics
from pvanalytics.quality.outliers import zscore
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

# %%
# First, we read in the ac_power_inv_7539 example, and visualize the min-max
# normalized time series.

pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
ac_power_file_1 = pvanalytics_dir / 'data' / 'ac_power_inv_7539.csv'
data = pd.read_csv(ac_power_file_1, index_col=0, parse_dates=True)

data['value_normalized'].plot()
plt.xlabel("Date")
plt.ylabel("Normalized AC Power")
plt.tight_layout()
plt.show()

# %%
# Add some outliers to the time series, for
# :py:func:`pvanalytics.quality.outliers.zscore` to detect.
anomaly_dictionary = {35: data['value_normalized'][35]*1.5,
                      80: -.5,
                      160: -.4,
                      195: data['value_normalized'][195]*2,
                      200: data['value_normalized'][200]*.1,
                      333:  data['value_normalized'][333]*2
                      }
data.loc[:, 'anomaly'] = False
# Create fake anomaly values based on anomaly_dictionary
for index, anomaly_value in anomaly_dictionary.items():
    index_date = data.iloc[index].name
    data.loc[index_date, 'value_normalized'] = anomaly_value
    data.loc[index_date, 'anomaly'] = True

data['value_normalized'].plot()
data.loc[data['anomaly'], 'value_normalized'].plot(ls='', marker='o')
plt.legend(labels=["AC Power", "Generated Outlier"])
plt.xlabel("Date")
plt.ylabel("Normalized AC Power")
plt.tight_layout()
plt.show()

# %%
# Use :py:func:`pvanalytics.quality.outliers.zscore` to identify
# outliers in the time series. Re-plot the data subset with this mask.
zscore_outlier_mask = zscore(data=data['value_normalized'])
data['value_normalized'].plot()
data.loc[zscore_outlier_mask, 'value_normalized'].plot(ls='', marker='o')
plt.legend(labels=["AC Power", "Detected Outlier"])
plt.xlabel("Date")
plt.ylabel("Normalized AC Power")
plt.tight_layout()
plt.show()
