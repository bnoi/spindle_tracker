import numpy as np


def bbox(points, shift=0):
    max_x, max_y = np.max(points, axis=0)
    min_x, min_y = np.min(points, axis=0)

    w = max_x - min_x
    h = max_y - min_y

    if shift > 0:
        min_x -= w * shift
        min_y -= h * shift
        max_x += w * shift
        max_y += h * shift

    x = min_x
    y = min_y
    w = max_x - min_x
    h = max_y - min_y

    return x, y, w, h
