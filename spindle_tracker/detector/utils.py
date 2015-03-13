import numpy as np


def create_log_kernel(radius, ndims, calibration):
    """
    """

    # Compute sigma
    sigma = radius / np.sqrt(ndims)
    sigma_pixels = [sigma / calibration[i] for i in range(ndims)]

    # Compute kernel size
    sizes = []
    middle = []
    for d in range(len(sigma_pixels)):
        hksizes = np.max([2, int((3 * sigma_pixels[d] + 0.5) + 1)])
        sizes.append(3 + 2 * hksizes)
        middle.append(1 + hksizes)

    middle = np.array(middle)

    # Init kernel
    kernel = np.zeros(sizes, dtype='float')

    # The gaussian normalization factor, divided by a constant value.
    # This is a fudge factor, that more or less put the quality values
    # close to the maximal value of a blob of optimal radius.
    C = 1 / 20 * (1 / sigma / np.sqrt(2 * np.pi)) ** ndims

    for i in range(sizes[0]):
        for j in range(sizes[1]):
            mantissa = 0
            exponent = 0
            coords = [i, j]
            for d in range(len(sigma_pixels)):
                x = calibration[d] * (coords[d] - middle[d])
                mantissa += -C * (x * x / sigma / sigma - 1)
                exponent += -x * x / 2 / sigma / sigma

            kernel[i, j] = mantissa * np.exp(exponent)

    # Vectorized version (10 time slower)
    # Investigate to use numba or cython
    # for coords in np.ndindex(kernel.shape):
    #    x = calibration * coords * middle
    #    mantissa = np.sum(-C * (x * x / sigma / sigma - 1))
    #    exponent = np.sum(-x * x / 2 / sigma / sigma)
    #
    #    kernel[coords] = mantissa * np.exp(exponent)

    return kernel
