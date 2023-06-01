"""
Detect if a System is Tracking
==============================

Identifying if a system is tracking or fixed tilt
"""

# %%
# It is valuable to identify if a system is fixed tilt or tracking for
# future analysis. This example shows how to use
# :py:func:`pvanalytics.system.is_tracking_envelope` to determine if a
# system is tracking or not by fitting data to a maximum power or
# irradiance envelope, and fitting this envelope to quadratic and
# quartic curves. The r^2 output from these fits is used to determine
# if the system fits a tracking or fixed-tilt profile.

import pvanalytics
from pvanalytics.system import is_tracking_envelope
from pvanalytics.features.clipping import geometric
from pvanalytics.features.daytime import power_or_irradiance
import pandas as pd
import pathlib
import matplotlib.pyplot as plt

# %%
# First, we import an AC power data stream from the SERF East site located at
# NREL. This data set is publicly available via the PVDAQ database in the
# DOE Open Energy Data Initiative (OEDI)
# (https://data.openei.org/submissions/4568), under system ID 50.
# This data is timezone-localized. This particular data stream is associated
# with a fixed-tilt system.

pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
ac_power_file = pvanalytics_dir / 'data' / \
    'serf_east_15min_ac_power.csv'
data = pd.read_csv(ac_power_file, index_col=0, parse_dates=True)
data = data.sort_index()
time_series = data['ac_power']
time_series = time_series.asfreq('15T')

# Plot the first few days of the time series to visualize it
time_series[:pd.to_datetime("2016-07-06 00:00:00-07:00")].plot()
plt.show()

# %%
# Run the clipping and the daytime filters on the time series.
# Both of these masks will be used as inputs to the
# :py:func:`pvanalytics.system.is_tracking_envelope` function.

# Generate the daylight mask for the AC power time series
daytime_mask = power_or_irradiance(time_series)

# Generate the clipping mask for the time series
clipping_mask = geometric(time_series)

# %%
# Now, we use :py:func:`pvanalytics.system.is_tracking_envelope` to
# identify if the data stream is associated with a tracking or fixed-tilt
# system.

predicted_mounting_config = is_tracking_envelope(time_series,
                                                 daytime_mask,
                                                 clipping_mask)

print("Estimated mounting configuration: " + predicted_mounting_config.name)
