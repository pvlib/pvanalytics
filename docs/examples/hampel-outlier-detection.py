"""
Hampel Outlier Detection
========================

Identifying outliers in time series using the :py:func:`pvanalytics.quality.outliers.hampel`
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
from pvanalytics.quality.outliers import tukey
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

# %%
# First, we read in the ac_power_inv_7539 example, and visualize a subset of the
# clipping periods via the "label" mask column.

pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
ac_power_file_1 = "C:/Users/kperry/Documents/source/repos/pvanalytics/pvanalytics/data/ac_power_inv_7539.csv"#pvanalytics_dir / 'data' / 'ac_power_inv_7539.csv'
data = pd.read_csv(ac_power_file_1, index_col=0, parse_dates=True)

data['value_normalized'].plot()
plt.xticks(rotation=20)
plt.xlabel("Date")
plt.ylabel("Normalized AC Power")
plt.tight_layout()
plt.show()


# %%
# Add some outliers to the time series, for :py:func:`pvanalytics.quality.outliers.tukey`
# to detect.



# %%
# Use :py:func:`pvanalytics.quality.outliers.tukey` to identify
# outliers in the time series. Re-plot the data subset with this mask.

data['value_normalized'].plot()
data.loc[predicted_clipping_mask, 'value_normalized'].plot(ls='', marker='o')
plt.legend(labels=["AC Power", "Detected Clipping"],
           title="Clipped")
plt.xticks(rotation=20)
plt.xlabel("Date")
plt.ylabel("Normalized AC Power")
plt.tight_layout()
plt.show()

