import numpy as np


def build_transformations_matrix(p1, p2):
    """
    """

    center = (p1 + p2) / 2

    # Setup vectors
    origin_vec = np.array([1, 0])
    current_vec = (center - p1).values[0]
    current_vec /= np.linalg.norm(current_vec)

    # Find the rotation angle
    cosa = np.dot(origin_vec, current_vec)
    cosa = np.abs(cosa) * -1
    theta = np.arccos(cosa)

    # Build rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]], dtype="float")

    # Build translation matrix
    T = np.array([[1, 0, -center['x']],
                  [0, 1, -center['y']],
                  [0, 0, 1]], dtype="float")

    # Make transformations from R and T in one
    A = np.dot(T.T, R)

    return A
