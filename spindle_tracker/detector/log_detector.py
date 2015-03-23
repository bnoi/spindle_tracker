import numpy as np
import pandas as pd

from scipy.ndimage import filters
from skimage.feature import peak_local_max

from .utils import create_log_kernel


def log_detector(source, radius, ndims, calibration):
    """
    Parameters
    ----------
    source : numpy.array
        Source image
    radius : float
        Typical radius of a spot (in image unit)
    ndims : int
        Dimensions of the image (should be 2 or 3)
    calibration : list of float
        Pixel sizes for each dimensions

    Examples
    --------

    source = TiffFile("/home/hadim/test.tif").asarray()

    # Parameters
    radius = 0.18
    ndims = 2
    calibration = np.array([0.0645, 0.0645])

    spots = log_detector(source, radius, ndims, calibration)

    # Take the 4 most intense peaks
    filtered_spots = spots.sort('quality', ascending=False).iloc[:4]

    # Plot
    plt.figure()
    plt.imshow(source, interpolation="none", aspect='equal')

    for i, (x, y, z, radius, quality) in filtered_spots.iterrows():
        plt.scatter(x, y, s=50, c='red', alpha=0.8)

    Return
    ------
    pandas.DataFrame. Each line represents a detected spots.
    """
    kernel = create_log_kernel(radius, ndims, calibration)

    fftconv = filters.convolve(source, kernel, mode='reflect', cval=0)

    peaks = peak_local_max(fftconv, min_distance=1, threshold_rel=0, indices=True, num_peaks=np.inf)

    spots = []
    for peak in peaks:
        x = peak[1]
        y = peak[0]
        if ndims < 3:
            z = 0
        else:
            z = peaks[2]

        quality = source[y, x]

        spots.append([x, y, z, radius, quality])

    spots = pd.DataFrame(spots, columns=['x', 'y', 'z', 'radius', 'quality'])

    return spots
