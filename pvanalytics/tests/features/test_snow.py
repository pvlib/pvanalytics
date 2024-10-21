import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal
from pvanalytics.features import snow


def test_get_irradiance_sapm():
    # solve Ee^2 + 2 Ee - 3 = 0
    c0, c1 = (2, 1)
    imp0 = 6.
    alpha_imp = -2. / 3
    temp_cell = 26.
    # effective irradiance = i_mp / factor
    i_mp = np.array([6., 0., np.nan])
    result = snow.get_irradiance_sapm(temp_cell, i_mp, imp0, c0, c1, alpha_imp)
    expected = np.array([1000., 0., np.nan])  # from suns to W/m2
    assert_array_equal(result, expected)
    # test with Series
    i_mp = pd.Series(data=i_mp)
    temp_cell = pd.Series(index=i_mp.index, data=26.)
    result = snow.get_irradiance_sapm(temp_cell, i_mp, imp0, c0, c1, alpha_imp)
    assert_series_equal(result, pd.Series(index=i_mp.index, data=expected))


def test_get_irradiance_imp():
    # solve Ee = Imp / Imp0
    imp0 = 6.
    i_mp = np.array([6., 0., np.nan])
    result = snow.get_irradiance_imp(i_mp, imp0)
    expected = np.array([1000., 0, np.nan])
    assert_array_equal(result, expected)
    # test with Series
    i_mp = pd.Series(data=i_mp)
    result = snow.get_irradiance_imp(i_mp, imp0)
    assert_series_equal(result, pd.Series(index=i_mp.index, data=expected))


def test_get_transmission():
    # solve modeled_ee / measured_ee, subject to i_mp>0, with bounds
    # T[T.isna()] = np.nan
    # T[i_mp == 0] = 0
    # T[T < 0] = np.nan
    # T[T > 1] = 1
    measured_ee = pd.Series(data=[1., 0.5, 1., -1., np.nan])
    modeled_ee = pd.Series(data=[1., 1., 1., 1., 1.])
    i_mp = pd.Series(data=[1., 1., 0., 1., 1.])
    result = snow.get_transmission(measured_ee, modeled_ee, i_mp)
    expected = pd.Series(data=[1., 1., 0., np.nan, np.nan])
    assert_series_equal(result, expected)


def test_categorize():
    measured_voltage = np.array([400., 450., 350., 420., 420., 495., 270.,
                                 200., 850., 0., 700.])
    modeled_voltage_with_calculated_transmission = np.array([np.nan, 500, 700,
                                                             700, 600, 550,
                                                             300, 200, 600,
                                                             0, 700])
    modeled_voltage_with_ideal_transmission = np.array([700, 600, 700, 750,
                                                        700, 700, 350, 250,
                                                        600, 400, 850])
    transmission = np.array([0.5, np.nan, 0.5, 0.9, 0.5, 0.9, 0.9, 0.9, 1, 0,
                             0.9])

    min_dcv = 300
    max_dcv = 800
    threshold_vratio = 0.7
    threshold_transmission = 0.6

    expected = np.array([None, None, 1, 2, 3, 4, 0, -1, -1, 0, -1])
    result, _ = snow.categorize(transmission, measured_voltage,
                                modeled_voltage_with_calculated_transmission,
                                modeled_voltage_with_ideal_transmission,
                                min_dcv, max_dcv, threshold_vratio,
                                threshold_transmission)
    assert_array_equal(result, expected)
