"""Tests for feature labeling functions."""
import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
import pvlib
from pvanalytics.features import clearsky


@pytest.mark.filterwarnings("ignore:Support for multi-dimensional indexing")
def test_reno_identical(quadratic):
    """Identical clearsky and measured irradiance all True"""
    index = pd.date_range(start='04/03/2020', freq='15min',
                          periods=len(quadratic))
    quadratic.index = index
    assert clearsky.reno(quadratic, quadratic).all()


@pytest.mark.filterwarnings("ignore:Support for multi-dimensional indexing")
@pytest.mark.filterwarnings("ignore:invalid value encountered in")
def test_reno_begining_end(quadratic):
    """clearsky conditions except in the middle of the dataset"""
    index = pd.date_range(start='03/03/2020', freq='15min',
                          periods=len(quadratic))
    quadratic.index = index
    ghi = quadratic.copy()
    ghi.iloc[20:31] *= 0.5
    ghi.iloc[31:41] = 0
    clear_times = clearsky.reno(ghi, quadratic, infer_limits=False)
    assert clear_times[0:20].all()
    assert not clear_times[21:40].any()
    assert clear_times[41:].all()


def test_reno_large_interval(quadratic):
    """clearsky.reno() raises ValueError if timestamp spacing too large."""
    index = pd.date_range(start='04/03/2020', freq='20min',
                          periods=len(quadratic))
    quadratic.index = index
    with pytest.raises(ValueError):
        clearsky.reno(quadratic, quadratic)


def test_reno_large_interval_infer_limits(quadratic):
    """clearsky.reno() raises ValueError if timestamp spacing too large
    for infer_limits=True."""
    index = pd.date_range(start='04/03/2020', freq='35min',
                          periods=len(quadratic))
    quadratic.index = index
    with pytest.raises(ValueError):
        clearsky.reno(quadratic, quadratic, infer_limits=True)


def test_reno_small_interval(quadratic):
    """clearsky.reno() raises ValueError if timestamp spacing too small."""
    index = pd.date_range(start='04/03/2020', freq='30s',
                          periods=len(quadratic))
    quadratic.index = index
    with pytest.raises(ValueError):
        clearsky.reno(quadratic, quadratic, infer_limits=True)


def test_reno_30min_interval(quadratic):
    """clearsky.reno() accepts data at 30 minute intervals."""
    index = pd.date_range(start='04/03/2020', freq='30min',
                          periods=len(quadratic))
    quadratic.index = index
    assert clearsky.reno(quadratic, quadratic, infer_limits=True).all()


def test_reno_infer_limits(quadratic):
    """clearsky.reno() uses thresholds inferred from the data interval."""
    index = pd.date_range(start='04/03/2020', freq='5min',
                          periods=len(quadratic))
    quadratic.index = index
    ghi = quadratic * (1 + 0.03 * np.sin(np.arange(len(quadratic))))
    expected = pvlib.clearsky.detect_clearsky(ghi, quadratic,
                                              infer_limits=True)
    assert_series_equal(
        clearsky.reno(ghi, quadratic, infer_limits=True), expected)
