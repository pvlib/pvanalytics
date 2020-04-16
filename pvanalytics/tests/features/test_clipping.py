"""Tests for features.clipping"""
import pytest
import pandas as pd
from pandas.util.testing import assert_series_equal
import numpy as np
from pvanalytics.features import clipping


@pytest.fixture
def quadratic():
    """Downward facing quadratic.

    Vertex at index 30 and roots at indices 0, 60.

    """
    q = -1000 * (np.linspace(-1, 1, 61) ** 2) + 1000
    return pd.Series(q)


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
