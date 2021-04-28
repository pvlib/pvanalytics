import numpy as np
import pandas as pd
from pvanalytics import metrics
import pytest


def test_performance_ratio_nrel():
    poa_global = np.array([921.75575, 916.11225, 914.8590833, 914.86375,
                           913.6426667, 889.6296667, 751.4611667])
    temp_air = np.array([28.89891667, 29.69258333, 30.21441667, 30.5815,
                         31.14808333, 31.5445, 31.63208333])
    wind_speed = np.array([1.796333333, 2.496916667, 2.413333333, 2.023666667,
                           1.844416667, 1.605583333, 1.18625])
    pac = np.array([39389.6625, 39425.3525, 39460.9625, 39125.315, 38888.59,
                    37520.8925, 27303.8025])
    pdc0 = 61056
    a = -3.58
    b = -0.113
    deltaT = 3
    gamma_pdc = -0.0028

    expected = 0.6873059
    performance_ratio = metrics.performance_ratio_nrel(poa_global, temp_air,
                                                       wind_speed, pac, pdc0,
                                                       a, b, deltaT, gamma_pdc)
    assert performance_ratio == pytest.approx(expected)


@pytest.fixture
def variability_inputs():
    times = pd.date_range('2019-01-01', freq='1min', periods=60*24)
    clear = pd.Series(100.0, index=times)
    # alternating sawtooth:
    jagged = pd.Series([100.0, 101.0] * (len(times)//2), index=times)
    return pd.DataFrame({
        'clear': clear,
        'jagged': jagged
    })


def test_variability_index(variability_inputs):
    # default freq parameter
    clear = variability_inputs['clear']
    jagged = variability_inputs['jagged']

    clear_clear = metrics.variability_index(clear, clear)
    assert clear_clear == pytest.approx(1.0)

    jagged_clear = metrics.variability_index(jagged, clear)
    assert jagged_clear == pytest.approx(2**0.5)


def test_variability_index_freq(variability_inputs):
    # custom freq parameter
    clear = variability_inputs['clear']
    jagged = variability_inputs['jagged']
    times = clear.resample('h').mean().index

    expected_clear_clear = pd.Series(1.0, index=times)
    actual_clear_clear = metrics.variability_index(clear, clear, freq='h')
    pd.testing.assert_series_equal(actual_clear_clear, expected_clear_clear)

    expected_jagged_clear = pd.Series(2**0.5, index=times)
    actual_jagged_clear = metrics.variability_index(jagged, clear, freq='h')
    pd.testing.assert_series_equal(actual_jagged_clear, expected_jagged_clear)
