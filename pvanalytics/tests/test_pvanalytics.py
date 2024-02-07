
from packaging.version import Version
import pvanalytics


def test___version__():
    # check that the version string is determined correctly.
    # if the version string is messed up for some reason, it should be
    # '0+unknown', which is not greater than '0.0.1'.
    version = Version(pvanalytics.__version__)
    assert version > Version('0.0.1')
