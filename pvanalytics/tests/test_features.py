"""Tests for feature labeling functions."""
import pytest
import pandas as pd
from pandas.util.testing import assert_series_equal
import numpy as np
from pvanalytics import features


@pytest.fixture
def quadratic():
    """A downward facing quadratic."""
    q = -1000 * (np.linspace(-1, 1, 60) ** 2) + 1000
    return pd.Series(q)


@pytest.fixture
def quadratic_clipped(quadratic):
    """A downward facing quadratic with clipping"""
    return np.minimum(quadratic, 800)


@pytest.fixture
def quadratic_compound(quadratic):
    """A sum of two quadratics."""
    return quadratic + quadratic


def test_clipping_levels(quadratic, quadratic_clipped):
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
        features.clipping_levels(
            quadratic_clipped, window=4,
            fraction_in_window=0.5, levels=4, rtol=5e-3)
    )


def test_clipping_levels_no_clipping(quadratic):
    """No clipping is identified in a data set that is jsut a quadratic."""
    assert not features.clipping_levels(
        quadratic, window=10, fraction_in_window=0.75, levels=4, rtol=5e-3
    ).any()


def test_clipping_levels_compound(quadratic):
    """No clipping is identified in the sum of two quadratics"""
    qsum = quadratic + quadratic
    assert not features.clipping_levels(
        qsum, window=10, fraction_in_window=0.75, levels=4, rtol=5e-3
    ).any()


def test_clipping_levels_compound_clipped(quadratic, quadratic_clipped):
    """Clipping is identified in summed quadratics when one quadratic has
    clipping."""
    assert features.clipping_levels(quadratic + quadratic_clipped).any()
