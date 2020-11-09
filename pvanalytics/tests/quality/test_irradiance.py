"""Tests for irradiance quality control functions."""
from datetime import datetime
import pytz
import pandas as pd
import numpy as np

import pytest
from pandas.util.testing import assert_series_equal

from pvanalytics.quality import irradiance


@pytest.fixture
def irradiance_qcrad():
    """Synthetic irradiance data and its expected quality flags.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    output = pd.DataFrame(
        columns=['ghi', 'dhi', 'dni', 'solar_zenith', 'dni_extra',
                 'ghi_limit_flag', 'dhi_limit_flag', 'dni_limit_flag',
                 'consistent_components', 'diffuse_ratio_limit'],
        data=np.array([[-100, 100, 100, 30, 1370, 0, 1, 1, 0, 0],
                       [100, -100, 100, 30, 1370, 1, 0, 1, 0, 0],
                       [100, 100, -100, 30, 1370, 1, 1, 0, 0, 1],
                       [1000, 100, 900, 0, 1370, 1, 1, 1, 1, 1],
                       [1000, 200, 800, 15, 1370, 1, 1, 1, 1, 1],
                       [1000, 200, 800, 60, 1370, 0, 1, 1, 0, 1],
                       [1000, 300, 850, 80, 1370, 0, 0, 1, 0, 1],
                       [1000, 500, 800, 90, 1370, 0, 0, 1, 0, 1],
                       [500, 100, 1100, 0, 1370, 1, 1, 1, 0, 1],
                       [1000, 300, 1200, 0, 1370, 1, 1, 1, 0, 1],
                       [500, 600, 100, 60, 1370, 1, 1, 1, 0, 0],
                       [500, 600, 400, 80, 1370, 0, 0, 1, 0, 0],
                       [500, 500, 300, 80, 1370, 0, 0, 1, 1, 1],
                       [0, 0, 0, 93, 1370, 1, 1, 1, 0, 0]]))
    dtypes = ['float64', 'float64', 'float64', 'float64', 'float64',
              'bool', 'bool', 'bool', 'bool', 'bool']
    for (col, typ) in zip(output.columns, dtypes):
        output[col] = output[col].astype(typ)
    return output


def test_daily_insolation_limits(albuquerque):
    """Daily insolation limits works with uniform timestamp spacing."""
    three_days = pd.date_range(
        start='1/1/2020',
        end='1/4/2020',
        closed='left',
        freq='H'
    )
    clearsky = albuquerque.get_clearsky(three_days, model='simplified_solis')
    assert irradiance.daily_insolation_limits(
        clearsky['ghi'], clearsky['ghi']
    ).all()
    assert not irradiance.daily_insolation_limits(
        pd.Series(0, index=three_days),
        clearsky['ghi']
    ).any()
    irrad = clearsky['ghi'].copy()
    irrad.loc['1/1/2020'] = irrad['1/1/2020']*0.7
    irrad.loc['1/2/2020'] = irrad['1/2/2020']*2.0
    irrad.loc['1/3/2020'] = irrad['1/3/2020']*0.3
    expected = pd.Series(False, irrad.index)
    expected.loc['1/1/2020'] = True
    assert_series_equal(
        expected,
        irradiance.daily_insolation_limits(irrad, clearsky['ghi']),
        check_names=False
    )


def test_daily_insolation_limits_uneven(albuquerque):
    """daily_insolation_limits works with uneven timestamp spacing."""
    three_days = pd.date_range(
        start='1/1/2020',
        end='1/4/2020',
        closed='left',
        freq='15T'
    )
    clearsky = albuquerque.get_clearsky(three_days, model='simplified_solis')
    ghi = clearsky['ghi'].copy()
    assert_series_equal(
        irradiance.daily_insolation_limits(ghi, ghi),
        pd.Series(True, three_days),
        check_names=False
    )
    ghi.loc[(ghi > 200) & (ghi < 202)] = np.nan
    ghi.loc[ghi == 0] = np.nan
    ghi.dropna(inplace=True)
    assert_series_equal(
        irradiance.daily_insolation_limits(
            ghi,
            albuquerque.get_clearsky(
                ghi.index,
                model='simplified_solis'
            )['ghi']
        ),
        pd.Series(True, index=ghi.index),
        check_names=False
    )
