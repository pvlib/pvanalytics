![lint and test](https://github.com/pvlib/pvanalytics/workflows/lint%20and%20test/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/pvlib/pvanalytics/badge.svg?branch=master)](https://coveralls.io/github/pvlib/pvanalytics?branch=master)

# PVAnalytics

PVAnalytics is a python library that supports analytics for PV
systems. It provides functions for quality control, filtering, and
feature labeling and other tools supporting the analysis of PV
system-level data.

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

  Other quality checks such as detecting stuck values, interpolation,
  and timestamp errors will also be included in `quality`.
* `filtering` as the name implies, contains functions for data
  filtering (e.g. day/night or solar position)
* `features` contains functons for identifying features in the data
  (e.g. inverter clipping, misaligned sensors, or clear sky
  conditions)
* `system` identification of PV system chatacteristics from data
  (e.g. nameplate power, orientation, azimuth)
* `translate` contains functions for translating data to other
  conditions (e.g. IV curve translators, temperature adjustment,
  irradiance adjustment)
* `metrics` contains functions for computing PV system-level metrics
* `fitting` contains submodules for different types of models that can
  be fit to data (e.g.  temperature models)
* `dataclasses` contains classes for normalizing data (e.g. an
  `IVCurve` class)
