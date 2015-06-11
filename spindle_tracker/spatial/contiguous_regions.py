import numpy as np
import pandas as pd


def contiguous_regions(condition):
    """
    Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index.
    Source : http://stackoverflow.com/a/4495197/458130
    """

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if type(condition) == pd.Series:

        if condition.iloc[0]:
            # If the start of condition is True prepend a 0
            idx = np.r_[0, idx]

        if condition.iloc[-1]:
            # If the end of condition is True, append the length of the array
            idx = np.r_[idx, condition.size] # Edit

    else:

        if condition[0]:
            # If the start of condition is True prepend a 0
            idx = np.r_[0, idx]

        if condition[-1]:
            # If the end of condition is True, append the length of the array
            idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx
