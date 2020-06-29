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

Gaps
----

Identify gaps in the data.

.. autosummary::
   :toctree: generated/

   quality.gaps.interpolation_diff
   quality.gaps.stale_values_diff

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

Features
========

Functions for detecting features in the data.

Clipping
--------

Functions for identifying inverter clipping

.. autosummary::
   :toctree: generated/

   features.clipping.levels

Clearsky
--------

.. autosummary::
   :toctree: generated/

   features.clearsky.reno

Orientation
-----------

Whether a system has a fixed orientation or is equipped with a tracker
can be determined by examining power or POA irradiance on days that
are relatively sunny. Two functions are provided that operate on power
or POA irradiance to identify system orientation on a daily
basis. These functions can tell you whether a day matches the profile
of a fixed system or system with a single-axis tracker; however, care
should be taken when interpreting negation of the output (i.e. looking
for days when the tracker is not functioning) since other factors such
as weather may interfere with classification.

To get a better understanding of when a single-axis is not working
(stuck in a fixed orientation) it may be possible to combine output
from both functions, using :py:func:`features.orientation.fixed_nrel`
to identify stuck days. Combining these fixed days with tracking days
using an exclusive or should give a reasonable estimate of relatively
sunny days where the tracker is stuck.

.. autosummary::
   :toctree: generated/

   features.orientation.fixed_nrel
   features.orientation.tracking_nrel

System
======

This module contains functions and classes for identifying system
characteristics.

.. autosummary::
   :toctree: generated/

   system.orientation

.. rubric:: References

.. [1]  C. N. Long and Y. Shi, An Automated Quality Assessment and Control
        Algorithm for Surface Radiation Measurements, The Open Atmospheric
        Science Journal 2, pp. 23-37, 2008.
