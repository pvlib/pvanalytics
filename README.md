![lint and test](https://github.com/pvlib/pvanalytics/workflows/lint%20and%20test/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/pvlib/pvanalytics/badge.svg?branch=master)](https://coveralls.io/github/pvlib/pvanalytics?branch=master)

# PVAnalytics

PVAnalytics is a python library that supports analytics for PV
systems. It provides functions for quality control, filtering, and
feature labeling and other tools supporting the analysis of PV
system-level data.

Documentation is available at 
[pvanalytics.readthedocs.io](https://pvanalytics.readthedocs.io).

## Library Overview

The functions provided by PVAnalytics are organized in modules based
on their anticipated use.  The structure/organization below is likely
to change as use cases are identified and refined and as package
content evolves.  The functions in `quality`, `filtering`, and
`features` will take a series of data and return a series of booleans.
* `quality` contains submodules for different kinds of data quality
  checks.
  * `irradiance` provides quality checks for irradiance
    measurements. This will initially contain an implementation of the
    QCRad algorithm, but any other quality tests for irradiance data
    should be added here.
  * `weather` has quality checks for weather data (for example tests
    for physically plausible values of temperature, wind speed,
    humidity, etc.)
  * `outliers` contains different functions for identifying outliers
    in the data.
  * `gaps` contains functions for identifying gaps in the data
    (i.e. missing values, stuck values, and interpolation).
  * `time` quality checks related to time (e.g. timestamp spacing)
  * `util` general purpose quality functions.

  Other quality checks such as detecting timestamp errors will also be
  included in `quality`.
* `filtering` as the name implies, contains functions for data
  filtering (e.g. day/night or solar position)
* `features` contains submodules with different methods for
  identifying and labeling salient features.
  * `clipping` functions for labeling inverter clipping.
  * `clearsky` functions for identifying periods of clear sky
    conditions.
* `system` identification of PV system characteristics from data
  (e.g. nameplate power, orientation, azimuth)
* `translate` contains functions for translating data to other
  conditions (e.g. IV curve translators, temperature adjustment,
  irradiance adjustment)
* `metrics` contains functions for computing PV system-level metrics
* `fitting` contains submodules for different types of models that can
  be fit to data (e.g.  temperature models)
* `dataclasses` contains classes for normalizing data (e.g. an
  `IVCurve` class)
