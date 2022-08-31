"""
Calculate Performance Ratio (NREL)
==================================

Calculate the NREL Performance Ratio for a system.
"""

# %%
# Identifying the NREL performance ratio (PR)

import pvanalytics
from pvanalytics.metrics import performance_ratio_nrel
import pandas as pd
import pathlib

# %%
# First, read in data from the NREL SERF East fixed-tilt system. This data
# set contains 15-minute interval AC power data.

pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
file = "C:/Users/kperry/Documents/source/repos/pvanalytics/pvanalytics/data/nrel_RSF_II.csv"#pvanalytics_dir / 'data' / 'serf_west_1min.csv'
data = pd.read_csv(file, index_col=0, parse_dates=True)


# %%
# Calculate the NREL performance ratio for the system, using the
# POA, ambient temperature, wind speed, and AC power fields, using the
# :py:func:`pvanalytics.metrics.performance_ratio_nrel` function.
pr = performance_ratio_nrel(data['poa_irradiance__1055'],
                            data['ambient_temp__1053'],
                            data['wind_speed__1051'],
                            data['inv1_ac_power_kw__1043'], 100) 

# %%
# 
print(pr)
