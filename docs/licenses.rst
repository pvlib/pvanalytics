Licenses
========

Solar Forecast Arbiter
----------------------

The functions and modules listed below are derived from
`solarforecastarbiter
<https://github.com/SolarArbiter/solarforecastarbiter-core>`_ under
the terms of the MIT License

  Copyright (c) 2019 SolarArbiter

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.

* The implementation of the QCRad algorithm in
  :py:mod:`pvanalytics.quality.irradiance`

* The interpolation and stuck value detection functions
  :py:func:`pvanalytics.quality.gaps.interpolation()` and
  :py:func:`pvanalytics.quality.gaps.stale_values()`

* Weather related quality functions

  * :py:func:`pvanalytics.quality.weather.temperature_limits`

  * :py:func:`pvanalytics.quality.weather.relative_humidity_limits`

  * :py:func:`pvanalytics.quality.weather.wind_limits`

* Level-based clipping detection in
  :py:func:`pvanalytics.features.clipping_levels`.
