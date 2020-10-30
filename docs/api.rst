.. currentmodule:: pvanalytics

#############
API Reference
#############

Quality
=======

Irradiance
----------

The ``check_*_limits_qcrad`` functions use the QCRad algorithm [1]_ to
identify irradiance measurements that are beyond physical limits.

.. autosummary::
   :toctree: generated/

   quality.irradiance.check_ghi_limits_qcrad
   quality.irradiance.check_dhi_limits_qcrad
   quality.irradiance.check_dni_limits_qcrad

All three checks can be combined into a single function call.

.. autosummary::
   :toctree: generated/

   quality.irradiance.check_irradiance_limits_qcrad

Irradiance measurements can also be checked for consistency.

.. autosummary::
   :toctree: generated/

   quality.irradiance.check_irradiance_consistency_qcrad

GHI and POA irradiance can be validated against clearsky values to
eliminate data that is unrealistically high.

.. autosummary::
   :toctree: generated/

   quality.irradiance.clearsky_limits

You may want to identify entire days that have unrealistically high or
low insolation. The following function examines daily insolation,
validating that it is within a reasonable range of the expected
clearsky insolation for the same day.

.. autosummary::
   :toctree: generated/

   quality.irradiance.daily_insolation_limits

Gaps
----

Identify gaps in the data.

.. autosummary::
   :toctree: generated/

   quality.gaps.interpolation_diff

Data sometimes contains sequences of values that are "stale" or
"stuck." These are contiguous spans of data where the value does not
change within the precision given. The functions below
can be used to detect stale values.

.. note::

   If the data has been altered in some way (i.e. temperature that has
   been rounded to an integer value) before being passed to these
   functions you may see unexpectedly large amounts of stale data.

.. autosummary::
   :toctree: generated/

   quality.gaps.stale_values_diff
   quality.gaps.stale_values_round

The following functions identify days with incomplete data.

.. autosummary::
   :toctree: generated/

   quality.gaps.completeness_score
   quality.gaps.complete

Many data sets may have leading and trailing periods of days with sporadic or
no data. The following functions can be used to remove those periods.

.. autosummary::
   :toctree: generated/

   quality.gaps.start_stop_dates
   quality.gaps.trim
   quality.gaps.trim_incomplete

Outliers
--------

Functions for detecting outliers.

.. autosummary::
   :toctree: generated/

   quality.outliers.tukey
   quality.outliers.zscore
   quality.outliers.hampel

Time
----

Quality control related to time. This includes things like time-stamp
spacing, time-shifts, and time zone validation.

.. autosummary::
   :toctree: generated/

   quality.time.spacing

Timestamp shifts, such as daylight savings, can be identified with
the following functions.

.. autosummary::
   :toctree: generated/

   quality.time.shifts_ruptures
   quality.time.has_dst

Utilities
---------

The :py:mod:`quality.util` module contains general-purpose/utility
functions for building your own quality checks.

.. autosummary::
   :toctree: generated/

   quality.util.check_limits
   quality.util.daily_min

Weather
-------

Quality checks for weather data.

.. autosummary::
   :toctree: generated/

   quality.weather.relative_humidity_limits
   quality.weather.temperature_limits
   quality.weather.wind_limits

In addition to validating temperature by comparing with limits, module
temperature should be positively correlated with irradiance. Poor
correlation could indicate that the sensor has become detached from
the module, for example. Unlike other functions in the
:py:mod:`quality` module which return Boolean masks over the input
series, this function returns a single Boolean value indicating
whether the entire series has passed (``True``) or failed (``False``)
the quality check.

.. autosummary::
   :toctree: generated/

   quality.weather.module_temperature_check

.. rubric:: References

.. [1]  C. N. Long and Y. Shi, An Automated Quality Assessment and Control
        Algorithm for Surface Radiation Measurements, The Open Atmospheric
        Science Journal 2, pp. 23-37, 2008.

Features
========

Functions for detecting features in the data.

Clipping
--------

Functions for identifying inverter clipping

.. autosummary::
   :toctree: generated/

   features.clipping.levels
   features.clipping.threshold

Clearsky
--------

.. autosummary::
   :toctree: generated/

   features.clearsky.reno

Orientation
-----------

System orientation refers to mounting type (fixed or tracker) and the
azimuth and tilt of the mounting. A system's orientation can be
determined by examining power or POA irradiance on days that are
relatively sunny.

This module provides functions that operate on power or POA irradiance
to identify system orientation on a daily basis. These functions can
tell you whether a day's profile matches that of a fixed system or
system with a single-axis tracker.

Care should be taken when interpreting function output since
other factors such as malfunctioning trackers can interfere with
identification.

.. autosummary::
   :toctree: generated/

   features.orientation.fixed_nrel
   features.orientation.tracking_nrel

Daytime
-------

Functions that return a Boolean mask indicating day and night.

.. autosummary::
   :toctree: generated/

   features.daytime.power_or_irradiance

System
======

This module contains functions and classes relating to PV system
parameters such as nameplate power, tilt, azimuth, or whether the
system is equipped with tracker.

Tracking
--------

.. autosummary::
   :toctree: generated/

   system.Tracker
   system.is_tracking_envelope

Orientation
-----------

The following function can be used to infer system orientation from
power or plane of array irradiance measurements.

.. autosummary::
   :toctree: generated/

   system.infer_orientation_daily_peak
   system.infer_orientation_fit_pvwatts

Metrics
=======

Performance Ratio
-----------------

The following functions can be used to calculate system performance metrics.

.. autosummary::
   :toctree: generated/

   metrics.performance_ratio_nrel