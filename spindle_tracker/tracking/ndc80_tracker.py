import logging
log = logging.getLogger(__name__)

import numpy as np

from sktracker.trajectories import Trajectories

from ..tracking import Tracker


class Ndc80Tracker(Tracker):

    MINIMUM_METADATA = ['SizeX', 'SizeY', 'SizeZ',
                        'PhysicalSizeX', 'PhysicalSizeY',
                        'TimeIncrement']

    def __init__(self, *args, **kwargs):
        """
        """

        super().__init__(*args, **kwargs)

    def kymo(self, var_name='trajs', marker='o', ls='-'):
        """
        """

        import matplotlib.pyplot as plt

        trajs = Trajectories(getattr(self, var_name))

        fig, ax = plt.subplots(nrows=1)
        ax = trajs.show(xaxis='t', yaxis='x_proj',
                        groupby_args={'level': "label"},
                        ax=ax, ls=ls, marker=marker)

        return fig

    def get_spb(self, traj_name='trajs', erase=False):
        """
        """

        trajs = Trajectories(getattr(self, traj_name)).copy()
        poles_idx = list(trajs.get_longest_segments(2))

        if hasattr(self, 'trajs_poles') and getattr(self, 'trajs_poles') is not None and not erase:
            return poles_idx

        if len(poles_idx) < 2:
            raise ValueError("Not enough segments to get poles")

        trajs['id'] = None
        trajs['side'] = None
        trajs = trajs.swaplevel("label", "t_stamp")
        trajs.sort_index(inplace=True)

        for i, l in zip(poles_idx, ['A', 'B']):
            trajs.loc[i, 'id'] = 'pole'
            trajs.loc[i, 'side'] = l

        trajs = trajs.swaplevel("label", "t_stamp")
        trajs.sort_index(inplace=True)

        self.save(trajs.copy(), 'trajs_poles')

        return poles_idx

    def interp(self, traj_name='trajs_gp'):
        """
        """

        trajs = getattr(self, traj_name)

        trajs = Trajectories(trajs.time_interpolate(coords=['x', 'y', 'z', 'I', 'w']))
        trajs.drop(list(filter(lambda x: x.startswith('v_') or x.startswith('a_'), trajs.columns)),
                   axis=1, inplace=True)

        self.save(trajs, 'trajs_interp')

    def project(self, poles_idx, traj_name='trajs_interp', progress=False):
        """
        """

        trajs = getattr(self, traj_name)
        trajs = trajs.project(poles_idx,
                              keep_first_time=False,
                              reference=None,
                              inplace=False,
                              progress=progress)

        self.save(trajs, 'trajs_poles')

    def guess_n_kts(self, traj_name='trajs_poles'):
        """
        """
        trajs = getattr(self, traj_name)

        def guess_n_kts(peaks):
            peaks['n_kts'] = np.round(peaks['I'] * 8 / peaks['I'].sum()).values
            return peaks

        trajs = trajs.groupby(level='t_stamp').apply(guess_n_kts)

        setattr(self, traj_name, Trajectories(trajs))
