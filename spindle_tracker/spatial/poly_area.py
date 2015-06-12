import numpy as np


def poly_area(xx, yy):
    return 0.5 * np.abs(np.dot(xx, np.roll(yy, 1)) - np.dot(yy, np.roll(xx, 1)))
