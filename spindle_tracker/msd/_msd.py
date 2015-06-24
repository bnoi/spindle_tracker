import numpy as np
import pandas as pd


def get_msd(traj, dt, label=None):

    shifts = np.arange(1, len(traj), dtype='int')
    msd = np.empty(shifts.shape, dtype='float')
    msd[:] = np.nan

    for i, shift in enumerate(shifts):
        d = traj[:-shift] - traj[shift:]
        d = d[~np.isnan(d).any(axis=1)]
        d = np.square(d).sum(axis=1)

        if len(d) > 0:
            msd[i] = np.mean(d)

    delays = shifts * dt

    msd = pd.DataFrame([delays, msd]).T
    msd.columns = ["delay", "msd"]

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
