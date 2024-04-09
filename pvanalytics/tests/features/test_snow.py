# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:39:12 2024

@author: cliff
"""
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pvanalytics.features import snow


def test_horizon_mask():
    horizon = pd.Series(index=range(0, 360), data=0)
    horizon[5:10] = 10
    result = snow.get_horizon_mask(horizon, azimuth=np.array([4, 5, 7]),
                                   elevation=np.array([-1, 11, np.nan]))
    expected = np.array([False, True, False])
    assert_array_equal(result, expected)


def test_categorize():
    vmp_ratio = np.array([np.nan, 0.9, 0.1, 0.6, 0.7, 0.9, 0.9])
    voltage = np.array([400., 400., 400., 400., 400., 400., 200.])
    transmission = np.array([0.5, np.nan, 0.9, 0.9, 0.5, 0.9, 0.9])
    min_dcv = 300
    threshold_vratio = 0.7
    threshold_transmission = 0.6
    # np.nan, vr<thres, vr<thres, vr=thres, vr>thres, vr>thres, vr<thres
    # vo>thres, vo>thres, vo>thres, vo>thres, vo>thres, vo>thres, vo<thres
    # tr<thres, np.nan, tr>thres, tr<thres, tr<thres, tr<thres, tr>thres
    expected = np.array([None, None, 2, 2, 2, 4, 0])
    result = snow.categorize(vmp_ratio, transmission, voltage, min_dcv,
                             threshold_vratio, threshold_transmission)
    assert_array_equal(result, expected)
