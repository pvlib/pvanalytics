"""
Calculate Performance Ratio (NREL)
==================================

Calculate the NREL Performance Ratio for a system.
"""

# %%
# Identifying the NREL performance ratio

import pvanalytics
from pvanalytics.metrics import performance_ratio_nrel
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

# %%
# First, read in data from the NREL SERF East fixed-tilt system. This data
# set contains 15-minute interval AC power data.

pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
file = pvanalytics_dir / 'data' / 'serf_east_15min_ac_power.csv'
data = pd.read_csv(file, index_col=0, parse_dates=True)


# %%
# Calculate the NREL performance ratio for the system, using the
# POA, ambient temperature, wind speed, and AC power fields, using the
# :py:func:`pvanalytics.features.metrics` function.
pr_time_series = performance_ratio_nrel(poa_global, temp_air, wind_speed,
                                        pac, pdc0) 

# %%
# Plot the AC power stream with the sunny day mask applied to it.

pr_time_series.plot()
plt.xlabel("Date")
plt.ylabel("NREL Performance Ratio")
plt.tight_layout()
plt.show()

