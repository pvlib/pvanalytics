"""
Infer System Orientation (Azimuth & Tilt) using Daily Peak
==========================================================

Infer the azimuth and tilt of a system using its daily peak value
"""

# %%
# Identifing and/or validating  the azimuth and tilt information for a
# system is important, as these values must be correct for future degradation
# and system yield analysis. This example shows how to use
# :py:func:`pvanalytics.system.infer_orientation_daily_peak` to estimate
# a system's azimuth and tilt, using the system's known latitude-longitude
# coordinates and an associated AC power time series.

import pvanalytics
from pvanalytics.features.daytime import power_or_irradiance
from pvanalytics.features.orientation import fixed_nrel
from pvanalytics import system as sys
import pandas as pd
import pathlib
import pvlib

# %%
# First, we import an AC power data stream from the SERF East site located at
# NREL. This data set is publicly available via the PVDAQ database in the
# DOE Open Energy Data Initiative (OEDI)
# (https://data.openei.org/submissions/4568). This data is timezone-localized.

pvanalytics_dir = pathlib.Path(pvanalytics.__file__).parent
ac_power_file = pvanalytics_dir / 'data' / \
    'serf_east_AC_power_system_estimate.csv'
data = pd.read_csv(ac_power_file, index_col=0, parse_dates=True)
data = data.sort_index()
time_series = data['ac_power']
time_series = time_series.asfreq('15T')

# Outline the ground truth metadata associated with the system
latitude = 39.742
longitude = -105.1727
actual_azimuth = 158
actual_tilt = 45

# %%
# Run the daytime and sunny day filters on the time series.
# Both of these masks will be used as inputs to the
# :py:func:`pvanalytics.system.infer_orientation_fit_pvwatts` function.

# Generate the daylight mask for the AC power time series
daytime_mask = power_or_irradiance(time_series)

# Get the sunny days associated with the system
sunny_days = fixed_nrel(time_series,
                        daytime_mask)

# Generate a list of possible tilts and azimuths to test the time series
# against
tilts = [*range(0, 65, 5)]
azimuths = [*range(90, 270, 5)]

# Get solar azimuth + zenith + ghi + dhi + dni from pvlib, based on
# lat-long coords
sun = pvlib.solarposition.get_solarposition(time_series.index,
                                            latitude,
                                            longitude)

# Get clear sky irradiance for the time series index
loc = pvlib.location.Location(latitude,
                              longitude)
CS = loc.get_clearsky(time_series.index)

# %%
# Run the pvlib data and the sensor-based time series data through
# the :py:func:`pvanalytics.system.infer_orientation_daily_peak` function.
best_azimuth, best_tilt = sys.infer_orientation_daily_peak(time_series,
                                                           sunny_days,
                                                           tilts,
                                                           azimuths,
                                                           sun.azimuth,
                                                           sun.zenith,
                                                           CS.ghi,
                                                           CS.dhi,
                                                           CS.dni)

# Compare actual system azimuth and tilt to predicted azimuth and tilt
print("Actual Azimuth: " + str(actual_azimuth))
print("Predicted Azimuth: " + str(best_azimuth))
print("Actual Tilt: " + str(actual_tilt))
print("Predicted Tilt: " + str(best_tilt))
