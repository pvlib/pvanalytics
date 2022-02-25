"""
Interpolated Data Periods
=========================

Identifying periods in a time series where the data has been
linearly interpolated.
"""

# %%
# Identifying periods where time series data has been linearly interpolated
# and removing these periods may help to reduce noise when performing future
# data analysis. This example shows how to use
# :py:func:`pvanalytics.quality.gaps.interpolation_diff`, which identifies and
# masks linearly interpolated periods.

import pvanalytics
from pvanalytics.quality import gaps
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

# %%
# First, we import the AC power data stream that we are going to check for
# interpolated periods. The time series we download is a normalized AC power
# time series from the PV Fleets Initiative, and is available via the DuraMAT
# DataHub:
# https://datahub.duramat.org/dataset/inverter-clipping-ml-training-set-real-data
pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
file = pvanalytics_dir / 'data' / 'ac_power_inv_2173.csv'
data = pd.read_csv(file, index_col=0, parse_dates=True)
data = data.asfreq("15T")

# %%
# We plot the time series before linearly interpolating missing data periods.
data.plot()
plt.xlabel("Date")
plt.ylabel("Normalized AC Power")
plt.tight_layout()
plt.show()

# %%
# We add linearly interpolated data periods to the time series for the
# :py:func:`pvanalytics.quality.gaps.interpolation_diff` to catch, and
# re-visualize the data with those interpolated periods masked.
interpolated_data_mask = data['value_normalized'].isna()
data = data.interpolate(method='linear', limit_direction='forward', axis=0)
data['value_normalized'].plot()
data.loc[interpolated_data_mask, "value_normalized"].plot(ls='', marker='.')
plt.legend(labels=["AC Power", "Interpolated Data"])
plt.xlabel("Date")
plt.ylabel("Normalized AC Power")
plt.tight_layout()
plt.show()

# %%
# Now, we use :py:func:`pvanalytics.quality.gaps.interpolation_diff` to
# identify linearly interpolated periods in the time series. We re-plot
# the data with this mask.
detected_interpolated_data_mask = gaps.interpolation_diff(
    data['value_normalized'])
data['value_normalized'].plot()
data.loc[detected_interpolated_data_mask,
         "value_normalized"].plot(ls='', marker='.')
plt.legend(labels=["AC Power", "Detected Interpolated Data"])
plt.xlabel("Date")
plt.ylabel("Normalized AC Power")
plt.tight_layout()
plt.show()
