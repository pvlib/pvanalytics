"""
Data Shift Detection & Removal
==============================
This example covers identifying data shift periods in a time series
and filtering them out, using the PVAnalytics quality.data_shifts
module.
"""

# %%
# This pipeline demonstrates the usage of the data shift algorithm on
# multiple simulated PVLib AC power data streams. It includes scenarios
# for data sets:
# -With a single changepoint in the series
# -With multiple changepoints in the series

import pvanalytics
import pandas as pd
import matplotlib.pyplot as plt
from pvanalytics.quality import data_shifts as dt
import pathlib

# %%
# For the first example, we are loading in a simulated PVLib AC power time
# series with a single changepoint, occurring on October 28, 2015.

pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
data_shift_file = pvanalytics_dir / 'data' / \
    'pvlib_data_shift_stream_example_1.csv'
df = pd.read_csv(data_shift_file)
df.index = pd.to_datetime(df['timestamp'])
df['value'].plot()
print("Changepoint at: " + str(df[df['label'] == 1].index[0]))

# %%
# Now you can run the data shift algorithm (with default parameters)
# on the data stream.  We re-plot the time series, with a vertical line
# where the detected changepoint is.

shift_mask = dt.detect_data_shifts(time_series=df['value'])
shift_list = list(df[shift_mask].index)
df['value'].plot()
for cpd in shift_list:
    plt.axvline(cpd, color="green")
plt.show()

# %%
# Filter the time series by detected changepoints, taking the longest
# continuous segment between detected changepoints.

interval_dict = dt.get_longest_shift_segment_dates(time_series=df['value'])
df['value'][interval_dict['start_date']:interval_dict['end_date']].plot()
plt.show()

# %%
# Now we re-run the pipeline using an example where there are multiple
# changepoints in the data stream.

data_shift_file = pvanalytics_dir / 'data' / \
    'pvlib_data_shift_stream_example_2.csv'
df = pd.read_csv(data_shift_file)
df.index = pd.to_datetime(df['timestamp'])
df['value'].plot()
print("Changepoints at: ")
print(str(list(df[df['label'] == 1].index)))

# %%
# Once again, we rerun the data shift algorithm (with default parameters)
# on the data stream, and plot the detected changepoints via green vertical
# line.
shift_mask = dt.detect_data_shifts(time_series=df['value'])
shift_list = list(df[shift_mask].index)
df['value'].plot()
for cpd in shift_list:
    plt.axvline(cpd, color="green")
plt.show()

# %%
# We then filter the time series by detected changepoints, taking the longest
# continuous segment between detected changepoints

interval_dict = dt.get_longest_shift_segment_dates(time_series=df['value'])
df['value'][interval_dict['start_date']:interval_dict['end_date']].plot()
plt.show()
