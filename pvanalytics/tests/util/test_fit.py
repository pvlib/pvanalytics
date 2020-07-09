"""Tests for curve fitting functions"""
import pytest
from pvanalytics.util import _fit


def test_fit_not_datetime_raises_error(quadratic):
    """Trying to fit to an integer index raises a ValueError."""
    with pytest.raises(ValueError):
        _fit.quadratic(quadratic)
