"""Functions for labeling shading and shodows."""
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import morphology, measure
import pvlib
from pvanalytics import util


def _to_image(data, width):
    """Convert data to an image.

    Parameters
    ----------
    data : Series
        Values of the pixels
    width : int
        Width of the image in pixels

    Returns
    -------
    ndarray
        Data reshaped into an ndarray with width `width`
    """
    return data.reshape(len(data) // width, width)


def _smooth(image):
    """Smooth `image` by local mean filtering.

    Local means are computed in a 3x3 square surrounding each pixel.
    """
    return ndimage.uniform_filter(image, size=15)


def _prepare_images(ghi, clearsky, daytime, interval):
    """Prepare data as images.

    Performs pre-processing steps on `ghi` and `clearsky` before
    returning images for use in the shadow detection algorithm.

    Returns
    -------
    ghi_image : np.ndarray
        Image form of `ghi`
    clearsky_image : np.ndarray
        Image form of `clearsky`
    clouds_image : np.ndarray
        Image of the cloudy periods in `ghi`
    image_times : pandas.DatetimeIndex
        Index for the data included in the returned images. Leading
        and trailing days with incomplete data are not included in the
        image, these times are needed to build a Series from the image
        later on.

    """
    # Fill missing days by interpolation. Interpolation is limited to
    # times with valid data before and after; missing data at the
    # beginning or end of the series is not filled in.
    image_width = 1440 // interval
    ghi = ghi.interpolate(limit_area='inside')
    # drop incomplete days.
    ghi = ghi[ghi.resample('D').transform('count') == image_width]
    image_times = ghi.index
    ghi_image = _to_image(ghi.to_numpy(), image_width)
    scaled_ghi = (ghi * 1000) / _smooth(ghi_image).max()
    scaled_clearsky = (clearsky * 1000) / clearsky.max()
    scaled_clearsky = scaled_clearsky.reindex_like(scaled_ghi)
    daytime = daytime.reindex_like(scaled_ghi)
    # Detect clouds.
    clouds = _detect_clouds(scaled_ghi, scaled_clearsky)
    # Interpolate across days (i.e. along columns) to remove clouds
    # replace clouds with nans
    #
    # This could probably be done directly with scipy.interpolate.inter1d,
    # but the easiest approach is to turn the image into a dataframe and
    # interpolate along the columns.
    cloudless_image = ghi_image.copy()
    cloudless_image[_to_image(clouds.to_numpy(), image_width)] = np.nan
    clouds_image = ghi_image.copy()
    clouds_image[~_to_image(clouds.to_numpy(), image_width)] = np.nan
    ghi_image = pd.DataFrame(cloudless_image).interpolate(
        axis=0,
        limit_direction='both'
    ).to_numpy()
    # set night to nan
    ghi_image[~_to_image(daytime.to_numpy(), image_width)] = np.nan
    return (
        ghi_image,
        _to_image(scaled_clearsky.to_numpy(), image_width),
        clouds_image,
        image_times
    )


def _detect_clouds(ghi, clearsky_ghi):
    """Use :py:func:`pvanalytics.clearsky.reno` to detect clouds"""
    clouds = pvlib.clearsky.detect_clearsky(
        ghi,
        clearsky_ghi,
        ghi.index,
        window_length=10
    )
    return clouds.rolling('60T').sum() == 0


def _remove_pillars(wires):
    j = np.array([[0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0]])
    k = np.array([[1, 0, 0, 0, 1],
                  [1, 0, 0, 0, 1],
                  [1, 0, 0, 0, 1]])
    a1 = np.logical_and(
        wires,
        np.logical_not(ndimage.binary_hit_or_miss(wires, j, k))
    )
    a2 = np.logical_and(
        wires,
        np.logical_not(
            ndimage.binary_hit_or_miss(wires, np.flipud(j), np.flipud(k))
        )
    )
    return np.logical_and(a1, a2)


def _fill_gaps(wires):
    """Fill 1-minute gaps."""
    mask = np.array([[1, 0, 1]])
    return np.logical_or(wires, ndimage.binary_hit_or_miss(wires, mask))


def _remove_spikes(wires):
    """Remove 1-day spikes."""
    mask = np.array([[0, 1, 0]]).T
    return np.logical_and(
        wires,
        np.logical_not(ndimage.binary_hit_or_miss(wires, mask))
    )


def _restore_gaps(wires):
    """Restore 2-minute and 1-minute gaps"""
    wires = np.logical_or(
        wires,
        ndimage.binary_hit_or_miss(
            wires,
            np.array([[0, 1, 0, 0, 1]]), np.array([[0, 0, 1, 1, 0]])
        )
    )
    return np.logical_or(
        wires,
        ndimage.binary_hit_or_miss(wires, np.array([[1, 0, 1]]))
    )


def _width(bbox):
    """Return the width of a bounding box.

    Parameters
    ----------
    bbox : tuple
        4-tuple specifying ``(min_row, min_col, max_row, max_col)``.

    Returns
    -------
    int
        The width of the bounding box in pixels.
    """
    return bbox[3] - bbox[1] + 1


def _height(bbox):
    """Return the height of a bounding box.

    Parameters
    ----------
    bbox : tuple
        4-tuple specifying ``(min_row, min_col, max_row, max_col)``.

    Returns
    -------
    int
        The height of the bounding box in pixels.
    """
    return bbox[2] - bbox[0] + 1


def _filter_blobs(wires, blob_length, connectivity):
    """Remove blobs shorter than blob_length.

    Examines all connected components in the image and removes any
    that span fewer than `blob_length` columns.
    """
    components = measure.label(wires, connectivity=connectivity, background=0)
    keep = [props.label for props in measure.regionprops(components)
            if _height(props.bbox) > blob_length]
    return np.isin(components, keep)


def _tand(degrees):
    """Tangent of an angle in degrees."""
    return np.tan(np.deg2rad(degrees))


def _filter_bars(wires):
    """Only keep shadows that are 'bar-shaped' and span multiple days."""
    for angle in range(0,90,5):
        if angle < 75:
            dim = 21
            mid = 10
        else:
            dim = round(10 + 2 * angle // 5)
            mid = round(6 + angle // 5)
        cc = np.arange(0, dim)
        se = np.zeros((dim, dim))
        if angle <= 45:
            rr = mid + np.round((cc - mid) * _tand(angle))
            se[cc, rr.astype('int')] = 1
        else:
            rr = mid + np.round((cc - mid) * _tand(90 - angle))
            se[cc, rr.astype('int')] = 1
        wires = np.logical_or(
            wires,
            morphology.binary_opening(wires, se.T)
        )
        wires = np.logical_or(
            wires,
            morphology.binary_opening(wires, np.flipud(se.T))
        )
    return wires


def _clean_wires(wires):
    """Clean up clouds that are connected to wires."""
    wires = _remove_pillars(wires)
    wires = _fill_gaps(wires)
    wires = _remove_spikes(wires)
    wires = _fill_gaps(wires)
    wires = _restore_gaps(wires)
    wires = _filter_blobs(wires, 20, connectivity=1)
    wires = _filter_bars(wires)
    wires = _filter_blobs(wires, 15, connectivity=2)
    return wires


def fixed(ghi, daytime, clearsky, interval=None):
    """Detects shadows form fixed structures such as wires and poles.

    Uses morphological image processing methods to identify shadows
    from fixed local objects in GHI data. GHI data are are assumed to
    be reasonably complete with relatively few missing values and at a
    fixed time interval nominally of 1 minute over the course of
    several months. Shadows are assumed to be relatively short
    duration. The algorithm forms a 2D image of the GHI data by
    arranging time of day along the x-axis and day of year along the
    y-axis. Rapid change in GHI in the x-direction is used to identify
    edges of shadows; continuity in the y-direction is used to
    separate local object shading from cloud shadows.

    Parameters
    ----------
    ghi : Series
        Time series of GHI measurements. Should be at 1-minute frequency
        for best results. Must be 1 year of data.
    daytime : Series
        Boolean series with True for times when the sun is up.
    clearsky : Series
        Clearsky GHI.
    interval : int, optional
        Interval between data points in minutes. If not specified the
        interval is inferred from the frequency of the index of `ghi`.

    Returns
    -------
    Series
        Boolean series with true for times that are impacted by shadows.

    References
    ----------
    References
    .. [1] Martin, C. E., Hansen, C. W., An Image Processing Algorithm to
       Identify Near-Field Shading in Irradiance Measurements, preprint 2016
    .. [2] Reno, M.J. and C.W. Hansen, "Identification of periods of clear sky
       irradiance in time series of GHI measurements" Renewable Energy, v90,
       p. 520-531, 2016.
    """
    if interval is None:
        interval = util.freq_to_timedelta(
            pd.infer_freq(ghi.index)
        ).seconds // 60
    # TODO Handle invalid intervals
    ghi_image, clearsky_image, clouds_image, index = _prepare_images(
        ghi,
        clearsky,
        daytime,
        interval
    )
    # normalize the GHI and dampen the dynamic range where the clear
    # sky model may have large errors (e.g. at very low sun elevation)
    alpha = 2000
    ghi_boosted = 1000 * (ghi_image + alpha) / (clearsky_image + alpha)

    # We must use scipy.ndimage here because skimage does not support
    # floating point the data outside the range [-1, 1]
    gradient = ndimage.morphological_gradient(ghi_boosted, size=(1, 3))
    min_gradient = 2  # TODO consider making this a parameter
    threshold = gradient > min_gradient  # binary image of wire candidates

    # From here we CAN use skimage we are working with binary
    # images. We use the grayscale version of these functions rather
    # than the binary version since the binary version do not support
    # the 'area_threshold' parameter.
    three_day_mask = morphology.rectangle(1, 3)
    wires = morphology.area_opening(
        morphology.closing(threshold, three_day_mask),
        area_threshold=200,
        connectivity=2  # all neighbors (including diagonals)
    )
    wires = _clean_wires(wires)
    # TODO convert wires back to a time-series indexed by ghi.index
    return wires, ghi_image, gradient
