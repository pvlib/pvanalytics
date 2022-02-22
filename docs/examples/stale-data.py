"""
Stale Data Filtering
===================

Identifying stale data in time series.
"""

# %%
# 

import pvanalytics
from pvanalytics.features import gaps
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import numpy as np

# %%
# 


# %%
# Now, use :py:func:`pvanalytics.features.gaps.stale_values_diff` to identify
# stale values in data.

gaps.stale_values_diff(x)


# %%
# Now, use :py:func:`pvanalytics.features.gaps.stale_values_round` to identify
# stale values in data.

gaps.stale_values_round(x)
