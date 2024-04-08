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
