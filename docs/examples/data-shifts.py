"""
Data Shift Detection & Filtering
================================
Identify data shifts/capacity changes in time series data
"""

# %%
# This example covers identifying data shifts/capacity changes in a time series
# and filtering the longest time series segment free of these shifts, using
# :py:func:`pvanalytics.quality.data_shifts.detect_data_shifts` and
# :py:func:`pvanalytics.quality.data_shifts.get_longest_shift_segment_dates`.

import pvanalytics
import pandas as pd
import matplotlib.pyplot as plt
from pvanalytics.quality import data_shifts as dt
import pathlib
import ruptures

# %%
# As an example, we load in a simulated PVLib AC power time series with a
# single changepoint, occurring on October 28, 2015.

pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
data_shift_file = pvanalytics_dir / 'data' / 'pvlib_data_shift.csv'
df = pd.read_csv(data_shift_file)
df.index = pd.to_datetime(df['timestamp'])
df['value'].plot()
print("Changepoint at: " + str(df[df['label'] == 1].index[0]))

# %%
# Now we run the data shift algorithm (with default parameters)
# on the data stream, using
# :py:func:`pvanalytics.quality.data_shifts.detect_data_shifts`. We re-plot
# the time series, with a vertical line where the detected changepoint is.

shift_mask = dt.detect_data_shifts(df['value'])
shift_list = list(df[shift_mask].index)
df['value'].plot()
for cpd in shift_list:
    plt.axvline(cpd, color="green")
plt.show()

# %%
# We filter the time series by detected changepoints, taking the longest
# continuous segment between detected changepoints, using
# :py:func:`pvanalytics.quality.data_shifts.get_longest_shift_segment_dates`.
# The filtered time series is then plotted.

start_date, end_date = dt.get_longest_shift_segment_dates(df['value'])
df['value'][start_date:end_date].plot()
plt.show()
