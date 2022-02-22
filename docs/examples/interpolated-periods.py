"""
Identifying Interpolated Data Periods
===================


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
# Now, use :py:func:`pvanalytics.features.gaps.interpolation_diff` to identify
# clipping periods in the time series. Re-plot the data subset with this mask.

gaps.interpolation_diff(x)
