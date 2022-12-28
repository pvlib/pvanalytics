"""Tests for feature labeling functions."""
import pytest
import pandas as pd
from pvanalytics.features import clearsky


@pytest.mark.filterwarnings("ignore:Support for multi-dimensional indexing")
def test_reno_identical(quadratic):
    """Identical clearsky and measured irradiance all True"""
    index = pd.date_range(start='04/03/2020', freq='15T',
                          periods=len(quadratic))
    quadratic.index = index
    assert clearsky.reno(quadratic, quadratic).all()


@pytest.mark.filterwarnings("ignore:Support for multi-dimensional indexing")
@pytest.mark.filterwarnings("ignore:invalid value encountered in")
def test_reno_begining_end(quadratic):
    """clearsky conditions except in the middle of the dataset"""
    index = pd.date_range(start='03/03/2020', freq='15T',
                          periods=len(quadratic))
    quadratic.index = index
    ghi = quadratic.copy()
    ghi.iloc[20:31] *= 0.5
    ghi.iloc[31:41] = 0
    clear_times = clearsky.reno(ghi, quadratic)
    assert clear_times[0:20].all()
    assert not clear_times[21:40].any()
    assert clear_times[41:].all()


def test_reno_large_interval(quadratic):
    """clearsky.reno() raises ValueError if timestamp spacing too large."""
    index = pd.date_range(start='04/03/2020', freq='20T',
                          periods=len(quadratic))
    quadratic.index = index
    with pytest.raises(ValueError):
        clearsky.reno(quadratic, quadratic)
