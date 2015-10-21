import numpy as np
import pandas as pd


def get_msd(traj, dt, with_nan=True, with_std=False, label=None):

    shifts = np.arange(1, len(traj), dtype='int')
    if with_std:
        msd = np.empty((len(shifts), 3), dtype='float')
    else:
        msd = np.empty((len(shifts), 2), dtype='float')
    msd[:] = np.nan

    msd[:, 1] = shifts * dt

    for i, shift in enumerate(shifts):
        diffs = traj[:-shift] - traj[shift:]
        if with_nan:
            diffs = diffs[~np.isnan(diffs).any(axis=1)]
        diffs = np.square(diffs).sum(axis=1)

        if len(diffs) > 0:
            msd[i, 0] = np.mean(diffs)

            if with_std:
                msd[i, 2] = np.std(diffs)


    msd = pd.DataFrame(msd)
    if with_std:
        msd.columns = ["msd", "delay", "std"]
    else:
        msd.columns = ["msd", "delay"]

    if label:
        msd['label'] = label
        msd.set_index(['label', 'delay'], drop=True, inplace=True)
    else:
        msd.set_index('delay', drop=True, inplace=True)

    msd.dropna(inplace=True)

    return msd


def get_homogenous_traj(trajs, side, coords=['x', 'y']):

    times = trajs.index.get_level_values('t_stamp').unique().astype('int')

    # Get the spindle center
    spbA = trajs.loc[pd.IndexSlice[:, 'spb', 'A'], coords].values
    spbB = trajs.loc[pd.IndexSlice[:, 'spb', 'B'], coords].values
    spindle_center = (spbA + spbB) / 2

    # Get KT
    traj = trajs.loc[pd.IndexSlice[:, 'kt', side], coords].reset_index(['main_label', 'side'],
                                                                       drop=True)
    traj -= spindle_center

    # Generate a single homogenous vector 'htraj' with nan values for missing timepoint
    htraj = np.empty((times[-1] + 1, traj.shape[1]))
    htraj[:] = np.nan
    htraj[times] = traj.values

    return htraj
