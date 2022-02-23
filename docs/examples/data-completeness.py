"""
Time Series Completeness
===================

Identifying days with missing data using a "completeness" score metric,
and filtering out consecutive days with a low completeness score for
future data analysis.
"""

# %%
# Identifying days with missing data and filtering these days out reduces noise
# when performing data analysis. This example shows how to use a
# daily data "completeness" score to identify and filter out days with missing
# data. This includes using
# :py:func:`pvanalytics.quality.gaps.completeness_score`,
# :py:func:`pvanalytics.quality.gaps.complete`, and
# :py:func:`pvanalytics.quality.gaps.trim_incomplete`.

import pvanalytics
from pvanalytics.quality import gaps
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

# %%
# First, we import the AC power data stream that we are going to check for
# completeness. The time series we download is a normalized AC power time
# series from the PV Fleets Initiative, and is available via the DuraMAT
# DataHub:
# https://datahub.duramat.org/dataset/inverter-clipping-ml-training-set-real-data

pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
file = pvanalytics_dir / 'data' / 'ac_power_inv_2173.csv'
data = pd.read_csv(file, index_col=0, parse_dates=True)
data = data.asfreq("15T")

# %%
# Now, use :py:func:`pvanalytics.quality.gaps.completeness_score` to get the
# percentage of daily data that isn't NaN.
data_completeness_score = gaps.completeness_score(data['value_normalized'])

# Visualize data completeness score as a time series.
data_completeness_score.plot()
plt.xlabel("Date")
plt.ylabel("Daily Completeness Score (Fractional)")
plt.tight_layout()
plt.show()

# %%
# Mask complete days, based on daily completeness score, using
# :py:func:`pvanalytics.quality.gaps.complete`
minimum_completeness = 0.333
daily_completeness_mask = gaps.complete(data['value_normalized'],
                                        minimum_completeness=0.333)

# Mask complete days, based on daily completeness score
data_completeness_score.plot()
data_completeness_score.loc[daily_completeness_mask].plot(ls='', marker='.')
data_completeness_score.loc[~daily_completeness_mask].plot(ls='', marker='.')
plt.axhline(y=minimum_completeness, color='r', linestyle='--')
plt.legend(labels=["Completeness Score", "Threshold met",
                   "Threshold not met", "Completeness Threshold (.33)"],
           loc="upper left")
plt.xlabel("Date")
plt.ylabel("Daily Completeness Score (Fractional)")
plt.tight_layout()
plt.show()

# %%
# Trim the time series based on the completeness score, where the time
# series must have at least 10 consecutive days of data that meet the
# completeness threshold. This is done using
# :py:func:`pvanalytics.quality.gaps.trim_incomplete`
number_consecutive_days = 10
completeness_trim_mask = gaps.trim_incomplete(data['value_normalized'],
                                              days=number_consecutive_days)
# Re-visualize the time series with the data masked by the trim mask
data[completeness_trim_mask]['value_normalized'].plot()
data[~completeness_trim_mask]['value_normalized'].plot()
plt.legend(labels=[True, False],
           title="Daily Data Passing")
plt.xlabel("Date")
plt.ylabel("Normalized AC Power")
plt.tight_layout()
plt.show()
