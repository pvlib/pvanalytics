"""
Time Series Completeness
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
# Now, use :py:func:`pvanalytics.features.gaps.completeness_score`

gaps.completeness_score(x)


# %%
# Get complete days, based on completeness score

gaps.complete(x)



# %%
# Trim the time series based on the completeness score
gaps.trim_incomplete(series, minimum_completeness=0.333333, days=10, freq=None)