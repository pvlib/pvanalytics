#!/usr/bin/env python

try:
    from setuptools import setup, find_packages
except ImportError:
    raise RuntimeError('setuptools is required')

DESCRIPTION = ('PVAnalytics is a python library for the analysis of ' +
               'photovoltaic system-level data.')

LONG_DESCRIPTION = """
PVAnalytics is a collection of functions for working with data
from photovoltaic power systems. The library includes functions for
general data quality tests such as outlier detection, validation that
data is physically plausible, filtering data for specific conditions,
and labeling specific features in the data.

Documentation: https://pvanalytics.readthedocs.io

Source code: https://github.com/pvlib/pvanalytics
"""

DISTNAME = 'pvanalytics'
MAINTAINER = "Will Vining"
MAINTAINER_EMAIL = 'wfvinin@sandia.gov'
LICENSE = 'MIT'
URL = 'https://github.com/pvlib/pvanalytics'

TESTS_REQUIRE = [
    'pytest',
]

INSTALL_REQUIRES = [
    'numpy >= 1.15.0',
    'pandas >= 0.23.0',
    'pvlib >= 0.8.0',
    'scipy >= 1.2.0',
    'statsmodels >= 0.9.0'
]

DOCS_REQUIRE = [
    'sphinx == 2.2.0'
]

EXTRAS_REQUIRE = {
    'optional': ['ruptures'],
    'test': TESTS_REQUIRE,
    'doc': DOCS_REQUIRE
}

EXTRAS_REQUIRE['all'] = sorted(set(sum(EXTRAS_REQUIRE.values(), [])))

SETUP_REQUIRES = ['setuptools_scm']

CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering'
]

PACKAGES = find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])

setup(
    name=DISTNAME,
    use_scm_version=True,
    packages=PACKAGES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    tests_require=TESTS_REQUIRE,
    setup_requires=SETUP_REQUIRES,
    ext_modules=[],
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    url=URL
)
