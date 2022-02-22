"""
Start-End Date Filtering
===================

Identifying start and end dates.
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
gaps.start_stop_dates(series, days=10)

# %%
# Now, use :py:func:`pvanalytics.features.gaps.trim`
gaps.trim(series, days=10)
