"""Tests for features.clipping"""
import pytest
from pandas.util.testing import assert_series_equal
import numpy as np
import pandas as pd
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

    power = full_day_clipped
    power.append(full_day)
    power.append(full_day)
    power.append(full_day)

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

    power = full_day
    power.append(full_day * 1.3)
    power.append(full_day * 1.2)
    power.append(full_day * 1.1)

    power.index = pd.date_range(
        start='01/01/2020 00:00', freq='10T', periods=len(power)
    )

    clipped = clipping.threshold(power)

    assert not clipped.any()
