"""
Clipping Detection
===================

Identifying clipping periods using the PVAnalytics clipping module.
"""

# %%
# Identifying and removing clipping periods from AC power time series
# data aids in generating more accurate degradation analysis results,
# as using clipped data can lead to over-predicting degradation. In this
# example, we show how to use :py:func:`pvanalytics.features.clipping.geometric`
# to mask clipping periods in an AC power time series. We use a
# normalized time series example provided by the PV Fleets Initiative,
# where clipping periods are labeled as True, and non-clipping periods are
# labeled as False. This example is provided taken from the DuraMAT DataHub: 
# https://datahub.duramat.org/dataset/inverter-clipping-ml-training-set-real-data
    
import pvanalytics
from pvanalytics.features.clipping import geometric
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

# %%
# First, read in the ac_power_inv_2054 example, and visualize clipping
# periods via the "label" mask column.

pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
ac_power_file_1 = pvanalytics_dir / 'data' / 'ac_power_inv_2054.csv'
data = pd.read_csv(ac_power_file_1, index_col=0, parse_dates=True)

# %%
# Now, use :py:func:`pvanalytics.features.clipping.geometric` to identify
# clipping periods in the time series. Re-plot the data with this mask.


# %%
# Compare the filter results to the ground-truth labeled data side-by-side,
# and generate accuracy and F1-score metrics.



