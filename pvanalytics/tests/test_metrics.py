import numpy as np
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
