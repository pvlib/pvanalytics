.. PVAnalytics documentation master file, created by
   sphinx-quickstart on Tue Feb 18 11:16:33 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PVAnalytics
===========

PVAnalytics is a python library that supports analytics for PV
systems. It provides functions for quality control, filtering, and
feature labeling and other tools supporting the analysis of PV
system-level data.

The source code for PVAnalytics is hosted on `github
<https://github.com/pvlib/pvanalytics>`_.

Library Overview
----------------

The functions provided by PVAnalytics are organized in submodules based
on their anticipated use. The list below provides a general overview; however,
not all modules have functions at this time, see the API reference for current
library status.

- :py:mod:`quality` contains submodules for different kinds of data quality
  checks.

  - :py:mod:`quality.irradiance` contains quality checks for irradiance
    measurements.
  - :py:mod:`quality.weather` contains quality checks for weather data (e.g.
    tests for physically plausible values of temperature, wind speed,
    humidity).
  - :py:mod:`quality.outliers` contains functions for identifying outliers.
  - :py:mod:`quality.gaps` contains functions for identifying gaps in the data
    (i.e. missing values, stuck values, and interpolation).
  - :py:mod:`quality.time` quality checks related to time (e.g. timestamp
    spacing, time shifts).
  - :py:mod:`quality.util` general purpose quality functions (e.g. simple
    range checks).

- :py:mod:`filtering` as the name implies, contains functions for data
  filtering.
- :py:mod:`features` contains submodules with different methods for
  identifying and labeling salient features.

  - :py:mod:`features.clipping` functions for labeling inverter clipping.
  - :py:mod:`features.clearsky` functions for identifying periods of clear sky
    conditions.
  - :py:mod:`features.orientation` functions for identifying
    orientation-related features in the data (e.g. days where the data looks
    like there is a functioning tracker). These functions are distinct from the
    functions in the :py:mod:`system` module in that we are identifying
    features of data rather than properties of the system that produced the
    data.

- :py:mod:`system` identification of PV system characteristics from data
  (e.g. nameplate power, tilt, azimuth)
- :py:mod:`translate` contains functions for translating data to other
  conditions (e.g. IV curve translators, temperature adjustment,
  irradiance adjustment)
- :py:mod:`metrics` contains functions for computing PV system-level metrics
  (e.g. performance ratio)
- :py:mod:`fitting` contains submodules for different types of models that can
  be fit to data (e.g.  temperature models)
- :py:mod:`dataclasses` contains classes for normalizing data (e.g. an
  ``IVCurve`` class)

Dependencies
------------

This project follows the guidelines laid out in
`NEP-29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_.
It supports:

- All minor versions of Python released 42 months prior to the project,
  and at minimum the two latest minor versions.
- All minor versions of numpy released in the 24 months prior to the project,
  and at minimum the last three minor versions
- The latest release of `PVLib <https://pvlib-python.readthedocs.io>`_.

PVAnalytics depends on the following packages:

.. literalinclude:: ../requirements.txt

Contents
========

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
