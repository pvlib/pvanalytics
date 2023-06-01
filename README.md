![lint and test](https://github.com/pvlib/pvanalytics/workflows/lint%20and%20test/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/pvlib/pvanalytics/badge.svg?branch=main)](https://coveralls.io/github/pvlib/pvanalytics?branch=main)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6110569.svg)](https://doi.org/10.5281/zenodo.6110569)


# PVAnalytics

PVAnalytics is a python library that supports analytics for PV
systems. It provides functions for quality control, filtering, and
feature labeling and other tools supporting the analysis of PV
system-level data.

PVAnalytics is available at [PyPI](https://pypi.org/project/pvanalytics/)
and can be installed using `pip`:

    pip install pvanalytics

Documentation and example usage is available at 
[pvanalytics.readthedocs.io](https://pvanalytics.readthedocs.io).

## Library Overview

The functions provided by PVAnalytics are organized in modules based
on their anticipated use.  The structure/organization below is likely
to change as use cases are identified and refined and as package
content evolves.  The functions in `quality` and
`features` take a series of data and return a series of booleans.
For more detailed descriptions, see our
[API Reference](https://pvanalytics.readthedocs.io/en/stable/api.html).

* `quality` contains submodules for different kinds of data quality
  checks.
  * `data_shifts` contains quality checks for detecting and 
    isolating data shifts in PV time series data.
  * `irradiance` provides quality checks for irradiance
    measurements. 
  * `weather` has quality checks for weather data (for example tests
    for physically plausible values of temperature, wind speed,
    humidity, etc.)
  * `outliers` contains different functions for identifying outliers
    in the data.
  * `gaps` contains functions for identifying gaps in the data
    (i.e. missing values, stuck values, and interpolation).
  * `time` quality checks related to time (e.g. timestamp spacing)
  * `util` general purpose quality functions.

* `features` contains submodules with different methods for
  identifying and labeling salient features.
  * `clipping` functions for labeling inverter clipping.
  * `clearsky` functions for identifying periods of clear sky
    conditions.
  * `daytime` functions for for identifying periods of day and night.
  * `orientation` functions for labeling data as corresponding to
    a rotating solar tracker or a fixed tilt structure.
  * `shading` functions for identifying shadows.
* `system` identification of PV system characteristics from data
  (e.g. nameplate power, orientation, azimuth)
* `metrics` contains functions for computing PV system-level metrics
