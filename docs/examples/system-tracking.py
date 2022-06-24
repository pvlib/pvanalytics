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
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

# %%
# First, we import the AC power data stream that we are going to check the 
# mounting configuration for. This particular data stream is associated with
# a XXXX system.
# This data set has a Pandas DateTime index, with the min-max normalized
# AC power time series represented in the 'value_normalized' column.
# The data is sampled at 15-minute intervals.

pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
file = pvanalytics_dir / 'data' / 'ac_power_inv_2173_stale_data.csv'
data = pd.read_csv(file, index_col=0, parse_dates=True)
data = data.asfreq("15T")

# %%
# Run the clipping and the daytime filters on the time series. 
# Both of these masks will be used as inputs to the 
# :py:func:`pvanalytics.system.is_tracking_envelope` function. 

# Generate the daylight mask for the AC power time series
daytime_mask = power_or_irradiance(time_series)
# Get the sunny days associated with the system
sunny_days = fixed_nrel(time_series,
                        daytime_mask, r2_min=0.94,
                        min_hours=5, peak_min=None)

# %%
# Now, we use :py:func:`pvanalytics.system.is_tracking_envelope` to
# identify if the data stream is associated with a tracking system.

is_tracking_envelope(series, daytime, clipping)