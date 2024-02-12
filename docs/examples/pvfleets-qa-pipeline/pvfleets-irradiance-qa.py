"""
PV Fleets QA Process: Irradiance
================================

PV Fleets Irradiance QA Pipeline
"""

# %%
# The NREL PV Fleets Data Initiative uses PVAnalytics routines to assess the
# quality of systems' PV data. In this example, the PV Fleets process for
# assessing the data quality of an irradiance data stream is shown. This
# example pipeline illustrates how several PVAnalytics functions can be used
# in sequence to assess the quality of an irradiance data stream.

import pandas as pd
import pathlib
from matplotlib import pyplot as plt
import pvanalytics
import pvlib
from pvanalytics.quality import data_shifts as ds
from pvanalytics.quality import gaps
from pvanalytics.quality.outliers import zscore
from pvanalytics.features.daytime import power_or_irradiance
from pvanalytics.quality.time import shifts_ruptures
from pvanalytics.features import daytime

# %%
# First, we import a POA irradiance data stream from a PV installation
# at NREL. This data set is publicly available via the PVDAQ database in the
# DOE Open Energy Data Initiative (OEDI)
# (https://data.openei.org/submissions/4568), under system ID 15.
# This data is timezone-localized.

pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
file = pvanalytics_dir / 'data' / 'system_15_poa_irradiance.parquet'
time_series = pd.read_parquet(file)
time_series.set_index('measured_on', inplace=True)
time_series.index = pd.to_datetime(time_series.index)
time_series = time_series['poa_irradiance__484']
latitude = 39.7406
longitude = -105.1775
data_freq = '15min'
time_series = time_series.asfreq(data_freq)

# %%
# First, let's visualize the original time series as reference.

time_series.plot(title="Original Time Series")
plt.xlabel("Date")
plt.ylabel("Irradiance, W/m^2")
plt.tight_layout()
plt.show()

# %%
# Now, let's run basic data checks to identify stale and abnormal/outlier
# data in the time series. Basic data checks include the following steps:
#
# 1) Flatlined/stale data periods
#    (:py:func:`pvanalytics.quality.gaps.stale_values_round`)
# 2) Negative irradiance data
# 3) "Abnormal" data periods, which are defined as days with a daily minimum
#    greater than 50 OR any data greater than 1300
# 4) Outliers, which are defined as more than one 4 standard deviations
#    away from the mean (:py:func:`pvanalytics.quality.outliers.zscore`)

# REMOVE STALE DATA (that isn't during nighttime periods)
# Day/night mask
daytime_mask = power_or_irradiance(time_series)
# Stale data mask
stale_data_mask = gaps.stale_values_round(time_series,
                                          window=3,
                                          decimals=2)
stale_data_mask = stale_data_mask & daytime_mask

# REMOVE NEGATIVE DATA
negative_mask = (time_series < 0)

# FIND ABNORMAL PERIODS
daily_min = time_series.resample('D').min()
erroneous_mask = (daily_min > 50)
erroneous_mask = erroneous_mask.reindex(index=time_series.index,
                                        method='ffill',
                                        fill_value=False)

# Remove values greater than or equal to 1300
out_of_bounds_mask = (time_series >= 1300)

# FIND OUTLIERS (Z-SCORE FILTER)
zscore_outlier_mask = zscore(time_series,
                             zmax=4,
                             nan_policy='omit')

# Get the percentage of data flagged for each issue, so it can later be logged
pct_stale = round((len(time_series[
    stale_data_mask].dropna())/len(time_series.dropna())*100), 1)
pct_negative = round((len(time_series[
    negative_mask].dropna())/len(time_series.dropna())*100), 1)
pct_erroneous = round((len(time_series[
    erroneous_mask].dropna())/len(time_series.dropna())*100), 1)
pct_outlier = round((len(time_series[
    zscore_outlier_mask].dropna())/len(time_series.dropna())*100), 1)

# Visualize all of the time series issues (stale, abnormal, outlier, etc)
time_series.plot()
labels = ["Irradiance"]
if any(stale_data_mask):
    time_series.loc[stale_data_mask].plot(ls='', marker='o', color="green")
    labels.append("Stale")
if any(negative_mask):
    time_series.loc[negative_mask].plot(ls='', marker='o', color="orange")
    labels.append("Negative")
if any(erroneous_mask):
    time_series.loc[erroneous_mask].plot(ls='', marker='o', color="yellow")
    labels.append("Abnormal")
if any(out_of_bounds_mask):
    time_series.loc[out_of_bounds_mask].plot(ls='', marker='o', color="yellow")
    labels.append("Too High")
if any(zscore_outlier_mask):
    time_series.loc[zscore_outlier_mask].plot(
        ls='', marker='o', color="purple")
    labels.append("Outlier")
plt.legend(labels=labels)
plt.title("Time Series Labeled for Basic Issues")
plt.xlabel("Date")
plt.ylabel("Irradiance, W/m^2")
plt.tight_layout()
plt.show()


# %%
# Now, let's filter out any of the flagged data from the basic irradiance
# checks (stale or abnormal data). Then we can re-visualize the data
# post-filtering.

# Filter the time series, taking out all of the issues
issue_mask = ((~stale_data_mask) & (~negative_mask) & (~erroneous_mask) &
              (~out_of_bounds_mask) & (~zscore_outlier_mask))
time_series = time_series[issue_mask]
time_series = time_series.asfreq(data_freq)

# Visualize the time series post-filtering
time_series.plot(title="Time Series Post-Basic Data Filtering")
plt.xlabel("Date")
plt.ylabel("Irradiance, W/m^2")
plt.tight_layout()
plt.show()

# %%
# We filter the time series based on its daily completeness score. This
# filtering scheme requires at least 25% of data to be present for each day to
# be included. We further require at least 10 consecutive days meeting this
# 25% threshold to be included.

# Visualize daily data completeness
data_completeness_score = gaps.completeness_score(time_series)

# Visualize data completeness score as a time series.
data_completeness_score.plot()
plt.xlabel("Date")
plt.ylabel("Daily Completeness Score (Fractional)")
plt.axhline(y=0.25, color='r', linestyle='-',
            label='Daily Completeness Cutoff')
plt.legend()
plt.tight_layout()
plt.show()

# Trim the series based on daily completeness score
trim_series = pvanalytics.quality.gaps.trim_incomplete(
    time_series,
    minimum_completeness=.25,
    freq=data_freq)
first_valid_date, last_valid_date = \
    pvanalytics.quality.gaps.start_stop_dates(trim_series)
time_series = time_series[first_valid_date.tz_convert(time_series.index.tz):
                          last_valid_date.tz_convert(time_series.index.tz)]
time_series = time_series.asfreq(data_freq)

# %%
# Next, we check the time series for any time shifts, which may be caused by
# time drift or by incorrect time zone assignment. To do this, we compare
# the modelled midday time for the particular system location to its
# measured midday time. We use
# :py:func:`pvanalytics.quality.gaps.stale_values_round`) to determine the
# presence of time shifts in the series.

# Get the modeled sunrise and sunset time series based on the system's
# latitude-longitude coordinates
modeled_sunrise_sunset_df = pvlib.solarposition.sun_rise_set_transit_spa(
    time_series.index, latitude, longitude)

# Calculate the midday point between sunrise and sunset for each day
# in the modeled irradiance series
modeled_midday_series = modeled_sunrise_sunset_df['sunrise'] + \
    (modeled_sunrise_sunset_df['sunset'] -
     modeled_sunrise_sunset_df['sunrise']) / 2

# Run day-night mask on the irradiance time series
daytime_mask = power_or_irradiance(time_series,
                                   freq=data_freq,
                                   low_value_threshold=.005)

# Generate the sunrise, sunset, and halfway points for the data stream
sunrise_series = daytime.get_sunrise(daytime_mask)
sunset_series = daytime.get_sunset(daytime_mask)
midday_series = sunrise_series + ((sunset_series - sunrise_series)/2)

# Convert the midday and modeled midday series to daily values
midday_series_daily, modeled_midday_series_daily = (
    midday_series.resample('D').mean(),
    modeled_midday_series.resample('D').mean())

# Set midday value series as minutes since midnight, from midday datetime
# values
midday_series_daily = (midday_series_daily.dt.hour * 60 +
                       midday_series_daily.dt.minute +
                       midday_series_daily.dt.second / 60)
modeled_midday_series_daily = \
    (modeled_midday_series_daily.dt.hour * 60 +
     modeled_midday_series_daily.dt.minute +
     modeled_midday_series_daily.dt.second / 60)

# Estimate the time shifts by comparing the modelled midday point to the
# measured midday point.
is_shifted, time_shift_series = shifts_ruptures(modeled_midday_series_daily,
                                                midday_series_daily,
                                                period_min=15,
                                                shift_min=15,
                                                zscore_cutoff=1.5)

# Create a midday difference series between modeled and measured midday, to
# visualize time shifts. First, resample each time series to daily frequency,
# and compare the data stream's daily halfway point to the modeled halfway
# point
midday_diff_series = (modeled_midday_series.resample('D').mean() -
                      midday_series.resample('D').mean()
                      ).dt.total_seconds() / 60

# Generate boolean for detected time shifts
if any(time_shift_series != 0):
    time_shifts_detected = True
else:
    time_shifts_detected = False

# Build a list of time shifts for re-indexing. We choose to use dicts.
time_shift_series.index = pd.to_datetime(
    time_shift_series.index)
changepoints = (time_shift_series != time_shift_series.shift(1))
changepoints = changepoints[changepoints].index
changepoint_amts = pd.Series(time_shift_series.loc[changepoints])
time_shift_list = list()
for idx in range(len(changepoint_amts)):
    if idx < (len(changepoint_amts) - 1):
        time_shift_list.append({"datetime_start":
                                str(changepoint_amts.index[idx]),
                                "datetime_end":
                                    str(changepoint_amts.index[idx + 1]),
                                "time_shift": changepoint_amts[idx]})
    else:
        time_shift_list.append({"datetime_start":
                                str(changepoint_amts.index[idx]),
                                "datetime_end":
                                    str(time_shift_series.index.max()),
                                "time_shift": changepoint_amts[idx]})

# Correct any time shifts in the time series
new_index = pd.Series(time_series.index, index=time_series.index)
for i in time_shift_list:
    new_index[(time_series.index >= pd.to_datetime(i['datetime_start'])) &
              (time_series.index < pd.to_datetime(i['datetime_end']))] = \
        time_series.index + pd.Timedelta(minutes=i['time_shift'])
time_series.index = new_index

# Remove duplicated indices and sort the time series (just in case)
time_series = time_series[~time_series.index.duplicated(
    keep='first')].sort_index()

# Plot the difference between measured and modeled midday, as well as the
# CPD-estimated time shift series.
midday_diff_series.plot()
time_shift_series.plot()
plt.title("Midday Difference Time Shift Series")
plt.xlabel("Date")
plt.ylabel("Midday Difference (Modeled-Measured), Minutes")
plt.tight_layout()
plt.show()

# Plot the heatmap of the irradiance time series
plt.figure()
# Get time of day from the associated datetime column
time_of_day = pd.Series(time_series.index.hour +
                        time_series.index.minute/60,
                        index=time_series.index)
# Pivot the dataframe
dataframe = pd.DataFrame(pd.concat([time_series, time_of_day], axis=1))
dataframe.columns = ["values", 'time_of_day']
dataframe = dataframe.dropna()
dataframe_pivoted = dataframe.pivot_table(index='time_of_day',
                                          columns=dataframe.index.date,
                                          values="values")
plt.pcolormesh(dataframe_pivoted.columns,
               dataframe_pivoted.index,
               dataframe_pivoted,
               shading='auto')
plt.ylabel('Time of day [0-24]')
plt.xlabel('Date')
plt.xticks(rotation=60)
plt.title('Post-Correction Heatmap, Time of Day')
plt.colorbar()
plt.tight_layout()
plt.show()

# %%
# Next, we check the time series for any abrupt data shifts. We take the
# longest continuous part of the time series that is free of data shifts.
# We use :py:func:`pvanalytics.quality.data_shifts.detect_data_shifts` to
# detect data shifts in the time series.

# Resample the time series to daily mean
time_series_daily = time_series.resample('D').mean()
data_shift_start_date, data_shift_end_date = \
    ds.get_longest_shift_segment_dates(time_series_daily)
data_shift_period_length = (data_shift_end_date - data_shift_start_date).days

# Get the number of shift dates
data_shift_mask = ds.detect_data_shifts(time_series_daily)
# Get the shift dates
shift_dates = list(time_series_daily[data_shift_mask].index)
if len(shift_dates) > 0:
    shift_found = True
else:
    shift_found = False

# Visualize the time shifts for the daily time series
print("Shift Found:", shift_found)
edges = [time_series_daily.index[0]] + \
    shift_dates + [time_series_daily.index[-1]]
fig, ax = plt.subplots()
for (st, ed) in zip(edges[:-1], edges[1:]):
    ax.plot(time_series_daily.loc[st:ed])
plt.title("Daily Time Series Labeled for Data Shifts")
plt.xlabel("Date")
plt.ylabel("Mean Daily Irradiance (W/m^2)")
plt.tight_layout()
plt.show()

# %%
# We filter the time series to only include the longest
# shift-free period.

# Filter the time series to only include the longest shift-free period
time_series = time_series[
    (time_series.index >= data_shift_start_date.tz_convert(
        time_series.index.tz)) &
    (time_series.index <= data_shift_end_date.tz_convert(
        time_series.index.tz))]

time_series = time_series.asfreq(data_freq)


# %%
# Display the final irradiance time series, post-QA filtering.
time_series.plot(title="Final Filtered Time Series")
plt.xlabel("Date")
plt.ylabel("Irradiance (W/m^2)")
plt.tight_layout()
plt.show()

# %%
# Generate a dictionary output for the QA assessment of this data stream,
# including the percent stale and erroneous data detected, any shift dates,
# and any detected time shifts.

qa_check_dict = {"original_time_zone_offset": time_series.index.tz,
                 "pct_stale": pct_stale,
                 "pct_negative": pct_negative,
                 "pct_erroneous": pct_erroneous,
                 "pct_outlier": pct_outlier,
                 "time_shifts_detected": time_shifts_detected,
                 "time_shift_list": time_shift_list,
                 "data_shifts": shift_found,
                 "shift_dates": shift_dates}

print("QA Results:")
print(qa_check_dict)
