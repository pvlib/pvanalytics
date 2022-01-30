"""Tests for features.clipping"""
import pytest
from pandas.util.testing import assert_series_equal
import numpy as np
import pandas as pd
from pvlib import irradiance, temperature, pvsystem, inverter
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from pvanalytics.features import clipping


@pytest.fixture
def quadratic_clipped(quadratic):
    """Downward facing quadratic with clipping at y=800"""
    return np.minimum(quadratic, 800)


def test_levels(quadratic, quadratic_clipped):
    """The clipped segment of a quadratic is properly identified."""
    expected = quadratic >= 800
    # because of the rolling window, the first clipped value in the
    # clipped series will not be marked as clipped (< 1/2 of the
    # values will be in a different level until there are at least two
    # clipped values in the window [for window=4])
    first_true = expected[quadratic >= 800].index[0]
    expected.loc[first_true] = False
    assert_series_equal(
        expected,
        clipping.levels(
            quadratic_clipped, window=4,
            fraction_in_window=0.5, levels=4, rtol=5e-3)
    )


def test_levels_no_clipping(quadratic):
    """No clipping is identified in a data set that is jsut a quadratic."""
    assert not clipping.levels(
        quadratic, window=10, fraction_in_window=0.75, levels=4, rtol=5e-3
    ).any()


def test_levels_compound(quadratic):
    """No clipping is identified in the sum of two quadratics"""
    qsum = quadratic + quadratic
    assert not clipping.levels(
        qsum, window=10, fraction_in_window=0.75, levels=4, rtol=5e-3
    ).any()


def test_levels_compound_clipped(quadratic, quadratic_clipped):
    """Clipping is identified in summed quadratics when one quadratic has
    clipping."""
    assert clipping.levels(quadratic + quadratic_clipped).any()


def test_levels_two_periods(quadratic, quadratic_clipped):
    """Two periods of clipping with lower values between them.

    The two periods of clipping should be flagged as clipping, and the
    central period of lower values should not be marked as clipping.
    """
    quadratic_clipped.loc[28:31] = [750, 725, 700, 650]
    clipped = clipping.levels(
        quadratic_clipped,
        window=4,
        fraction_in_window=0.5,
        levels=4,
        rtol=5e-3
    )
    assert not clipped[29:33].any()
    assert clipped[20:28].all()
    assert clipped[35:40].all()
    assert not clipped[0:10].any()
    assert not clipped[50:].any()


def test_levels_missing_data(quadratic, quadratic_clipped):
    quadratic[10:20] = np.nan
    quadratic_clipped[10:20] = np.nan
    assert_series_equal(
        pd.Series(False, quadratic.index),
        clipping.levels(quadratic, window=10)
    )
    assert not clipping.levels(quadratic_clipped, window=10)[10:20].any()


def test_threshold_no_clipping(quadratic):
    """In a data set with a single quadratic there is no clipping."""
    quadratic.index = pd.date_range(
        start='01/01/2020 07:30',
        freq='10T',
        periods=61
    )
    assert not clipping.threshold(quadratic).any()


def test_threshold_no_clipping_with_night(quadratic):
    """In a data set with a single quadratic surrounded by zeros there is
    no clipping."""
    quadratic.index = pd.date_range(
        start='01/01/2020 07:30',
        freq='10T',
        periods=61
    )
    full_day = quadratic.reindex(
        pd.date_range(
            start='01/01/2020 00:00',
            end='01/01/2020 23:50',
            freq='10T')
    )
    full_day.fillna(0)
    assert not clipping.threshold(quadratic).any()


def test_threshold_clipping(quadratic_clipped):
    """In a data set with a single clipped quadratic clipping is
    indicated."""
    quadratic_clipped.index = pd.date_range(
        start='01/01/2020 07:30',
        freq='10T',
        periods=61
    )
    assert not clipping.threshold(quadratic_clipped).all()
    assert clipping.threshold(quadratic_clipped).any()


def test_threshold_clipping_with_night(quadratic_clipped):
    """Clipping is identified in the daytime with periods of zero power
    before and after simulating night time conditions."""
    quadratic_clipped.index = pd.date_range(
        start='01/01/2020 07:30',
        freq='10T',
        periods=61
    )
    full_day = quadratic_clipped.reindex(
        pd.date_range(
            start='01/01/2020 00:00',
            end='01/01/2020 23:50',
            freq='10T')
    )
    full_day.fillna(0)
    assert not clipping.threshold(full_day).all()
    assert clipping.threshold(full_day)[quadratic_clipped.index].any()


def test_threshold_clipping_with_freq(quadratic_clipped):
    """Passing the frequency gives same result as infered frequency."""
    quadratic_clipped.index = pd.date_range(
        start='01/01/2020 07:30',
        freq='10T',
        periods=61
    )
    assert_series_equal(
        clipping.threshold(quadratic_clipped),
        clipping.threshold(quadratic_clipped, freq='10T')
    )


def test_threshold_clipping_with_interruption(quadratic_clipped):
    """Test threshold clipping with period of no clipping mid-day."""
    quadratic_clipped.loc[28:31] = [750, 725, 700, 650]
    quadratic_clipped.index = pd.date_range(
        start='01/01/2020 07:30',
        freq='10T',
        periods=61
    )
    clipped = clipping.threshold(quadratic_clipped)

    assert not clipped.iloc[0:10].any()
    assert not clipped.iloc[28:31].any()
    assert not clipped.iloc[50:].any()
    assert clipped.iloc[17:27].all()
    assert clipped.iloc[32:40].all()


def test_threshold_clipping_four_days(quadratic, quadratic_clipped):
    """Clipping is identified in the first of four days."""
    quadratic.index = pd.date_range(
        start='01/01/2020 07:30',
        freq='10T',
        periods=61
    )
    quadratic_clipped.index = pd.date_range(
        start='01/01/2020 07:30',
        freq='10T',
        periods=61
    )
    full_day_clipped = quadratic_clipped.reindex(
        pd.date_range(
            start='01/01/2020 00:00',
            end='01/01/2020 23:50',
            freq='10T')
    )
    full_day = quadratic.reindex(
        pd.date_range(
            start='01/01/2020 00:00',
            end='01/01/2020 23:50',
            freq='10T')
    )
    full_day_clipped.fillna(0)
    full_day.fillna(0)

    # scale the rest of the days below the clipping threshold
    full_day *= 0.75

    power = pd.concat([
        full_day_clipped,
        full_day,
        full_day,
        full_day,
    ])

    power.index = pd.date_range(
        start='01/01/2020 00:00', freq='10T', periods=len(power)
    )

    clipped = clipping.threshold(power)

    assert clipped['01/01/2020'].any()
    assert not clipped['01/02/2020':].any()


def test_threshold_no_clipping_four_days(quadratic):
    """Four days with no clipping."""
    quadratic.index = pd.date_range(
        start='01/01/2020 07:30',
        freq='10T',
        periods=61
    )
    full_day = quadratic.reindex(
        pd.date_range(
            start='01/01/2020 00:00',
            end='01/01/2020 23:50',
            freq='10T')
    )
    full_day.fillna(0)

    power = pd.concat([
        full_day,
        full_day * 1.3,
        full_day * 1.2,
        full_day * 1.1,
    ])

    power.index = pd.date_range(
        start='01/01/2020 00:00', freq='10T', periods=len(power)
    )

    clipped = clipping.threshold(power)

    assert not clipped.any()


@pytest.fixture(scope='module')
def july():
    return pd.date_range(start='7/1/2020', end='8/1/2020', freq='T')


@pytest.fixture(scope='module')
def clearsky_july(july, albuquerque):
    return albuquerque.get_clearsky(
        july,
        model='simplified_solis'
    )


@pytest.fixture(scope='module')
def solarposition_july(july, albuquerque):
    return albuquerque.get_solarposition(july)


@pytest.fixture
def power_pvwatts(request, clearsky_july, solarposition_july):
    pdc0 = 100
    pdc0_inverter = 110
    tilt = 30
    azimuth = 180
    pdc0_marker = request.node.get_closest_marker("pdc0_inverter")
    if pdc0_marker is not None:
        pdc0_inverter = pdc0_marker.args[0]
    poa = irradiance.get_total_irradiance(
        tilt, azimuth,
        solarposition_july['zenith'], solarposition_july['azimuth'],
        **clearsky_july
    )
    cell_temp = temperature.sapm_cell(
        poa['poa_global'], 25, 0,
        **TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    )
    dc = pvsystem.pvwatts_dc(poa['poa_global'], cell_temp, pdc0, -0.004)
    return inverter.pvwatts(dc, pdc0_inverter)


@pytest.mark.parametrize('freq', ['T', '5T', '15T', '30T', 'H'])
def test_geometric_no_clipping(power_pvwatts, freq):
    clipped = clipping.geometric(power_pvwatts.resample(freq).asfreq())
    assert not clipped.any()


@pytest.mark.pdc0_inverter(60)
@pytest.mark.parametrize('freq', ['T', '5T', '15T', '30T', 'H'])
def test_geometric_clipping(power_pvwatts, freq):
    clipped = clipping.geometric(power_pvwatts.resample(freq).asfreq())
    assert clipped.any()


@pytest.mark.pdc0_inverter(65)
@pytest.mark.parametrize('freq', ['5T', '15T', '30T', 'H'])
def test_geometric_clipping_correct(power_pvwatts, freq):
    power = power_pvwatts.resample(freq).asfreq()
    clipped = clipping.geometric(power)
    expected = power == power.max()
    if freq == '5T':
        assert np.allclose(power[clipped], power.max(), atol=0.5)
    else:
        assert_series_equal(clipped, expected)


@pytest.mark.pdc0_inverter(65)
def test_geometric_clipping_midday_clouds(power_pvwatts):
    power = power_pvwatts.resample('15T').asfreq()
    power.loc[power.between_time(
        start_time='17:30', end_time='19:30',
        include_start=True, include_end=True
    ).index] = list(range(30, 39)) * 31
    clipped = clipping.geometric(power)
    expected = power == power.max()
    assert_series_equal(clipped, expected)


@pytest.mark.pdc0_inverter(80)
def test_geometric_clipping_window(power_pvwatts):
    power = power_pvwatts.resample('15T').asfreq()
    clipped = clipping.geometric(power)
    assert clipped.any()
    clipped_window = clipping.geometric(power, window=24)
    assert not clipped_window.any()


@pytest.mark.pdc0_inverter(89)
def test_geometric_clipping_tracking(power_pvwatts):
    power = power_pvwatts.resample('15T').asfreq()
    clipped = clipping.geometric(power)
    assert clipped.any()
    clipped = clipping.geometric(power, tracking=True)
    assert not clipped.any()


@pytest.mark.pdc0_inverter(80)
def test_geometric_clipping_window_overrides_tracking(power_pvwatts):
    power = power_pvwatts.resample('15T').asfreq()
    clipped = clipping.geometric(power, tracking=True)
    assert clipped.any()
    clipped_override = clipping.geometric(power, tracking=True, window=24)
    assert not clipped_override.any()


@pytest.mark.parametrize('freq', ['5T', '15T'])
def test_geometric_clipping_missing_data(freq, power_pvwatts):
    power = power_pvwatts.resample(freq).asfreq()
    power.loc[power.between_time('09:00', '10:30').index] = np.nan
    power.loc[power.between_time('12:15', '12:45').index] = np.nan
    power.dropna(inplace=True)
    with pytest.raises(ValueError,
                       match="Cannot infer frequency of `ac_power`. "
                             "Please resample or pass `freq`."):
        clipping.geometric(power)
    assert not clipping.geometric(power, freq=freq).any()


def test_geometric_index_not_sorted():
    power = pd.Series(
        [1, 2, 3],
        index=pd.DatetimeIndex(
            ['20200201 0700', '20200201 0630', '20200201 0730']
        )
    )
    with pytest.raises(ValueError,
                       match=r"Index must be monotonically increasing\."):
        clipping.geometric(power, freq='30T')
