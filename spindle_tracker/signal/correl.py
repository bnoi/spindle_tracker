import numpy as np

__all__ = ["get_autocorrel"]


def get_autocorrel(x):
    xc = np.correlate(x, x, mode='full')
    xc /= xc[xc.argmax()]
    xchalf = xc[xc.size / 2:]
    return xchalf

