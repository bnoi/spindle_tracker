import itertools
import logging
log = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from sktracker.trajectories import Trajectories
from sktracker.tracker.solver import ByFrameSolver
from sktracker.tracker.solver import GapCloseSolver

from ..tracking import Tracker


class Ndc80Tracker(Tracker):

    MINIMUM_METADATA = ['SizeX', 'SizeY', 'SizeZ',
                        'PhysicalSizeX', 'PhysicalSizeY',
                        'TimeIncrement']

    def __init__(self, *args, **kwargs):
        """
        """

        super().__init__(*args, **kwargs)

    def track(self, trajs):
        """
        """

        coords = ['x', 'y']
        max_speed = 0.1
        penalty = 1.05

        parameters_brownian = {'max_speed': max_speed,
                               'coords': coords,
                               'penalty': penalty}

        parameters_directed = {'max_speed': max_speed,
                               'past_traj_time': 10,
                               'smooth_factor': 0,
                               'interpolation_order': 1,
                               'coords': coords,
                               'penalty': penalty}

        #solver = ByFrameSolver.for_directed_motion(trajs, **parameters_directed)
        solver = ByFrameSolver.for_brownian_motion(trajs, **parameters_brownian)

        trajs = solver.track(progress_bar=True)
        self.save(trajs.copy(), 'trajs')

        trajs = self.trajs.copy()
        maximum_gap = 50

        parameters_gap_close = {'max_speed': max_speed,
                                'maximum_gap': maximum_gap,
                                'use_t_stamp': False,
                                'link_percentile': 99,
                                'coords': coords}

        gc_solver = GapCloseSolver.for_brownian_motion(trajs, **parameters_gap_close)
        trajs = gc_solver.track()
        self.save(trajs, 'trajs_gp')

    def remove_small_segments(self, traj_name='trajs_gp', n=1):
        """Remove segments smaller or equal than "n"
        """

        trajs = getattr(self, traj_name)
        to_remove = filter(lambda x: len(x[1]) <= 1, trajs.segment_idxs.items())
        to_remove_idx = [x[1][0] for x in to_remove]
        to_remove_idx = list(itertools.chain(*to_remove_idx))
        trajs.remove_segments(to_remove_idx)

        log.info("{} small segments has been deleted.".format(len(to_remove_idx)))

    def interp(self, traj_name='trajs_gp'):
        """
        """

        log.info('Interpolate trajectories')
        trajs = getattr(self, traj_name)

        trajs = Trajectories(trajs.time_interpolate(coords=['x', 'y', 'z', 'I', 'w']))
        trajs.drop(list(filter(lambda x: x.startswith('v_') or x.startswith('a_'), trajs.columns)),
                   axis=1, inplace=True)

        self.save(trajs, 'trajs_interp')

    def get_spb(self, traj_name='trajs_interp', erase=False):
        """Find pairs of segments on which the mean distance on common timepoints is the bigger.
        """

        if hasattr(self, 'trajs_poles') and getattr(self, 'trajs_poles') is not None and not erase:
            trajs = getattr(self, 'trajs_poles')
            poleA = trajs[(trajs.id == 'pole') & (trajs.side == 'A')]
            poleA_id = poleA.index.get_level_values('label')[0]
            poleB = trajs[(trajs.id == 'pole') & (trajs.side == 'B')]
            poleB_id = poleB.index.get_level_values('label')[0]
            return (poleA_id, poleB_id)

        log.info("Find which segents are poles")

        trajs = Trajectories(getattr(self, traj_name)).copy()

        all_d = []
        ids = []
        for (id1, seg1), (id2, seg2) in itertools.combinations(trajs.iter_segments, 2):
            seg1 = seg1.loc[:, ['x', 'y', 't']].reset_index(drop=True)
            seg2 = seg2.loc[:, ['x', 'y', 't']].reset_index(drop=True)

            merged = pd.merge(seg1.reset_index(), seg2.reset_index(), on='t')

            if not merged.empty:
                p1 = merged.loc[:, ['x_x', 'y_x']].values
                p2 = merged.loc[:, ['x_y', 'y_y']].values
                vec = p1 - p2
                d = np.sqrt(vec[:, 0]**2 + vec[:, 1]**2)
                d_mean = d.mean()

                all_d.append(d_mean)
                ids.append((id1, id2))

        max_id = np.argmax(all_d)
        poles_idx = ids[max_id]

        # Label trajs correctly
        trajs['id'] = None
        trajs['side'] = None
        trajs = trajs.swaplevel("label", "t_stamp")
        trajs.sort_index(inplace=True)

        for i, l in zip(poles_idx, ['A', 'B']):
            trajs.loc[i, 'id'] = 'pole'
            trajs.loc[i, 'side'] = l

        trajs = trajs.swaplevel("label", "t_stamp")
        trajs.sort_index(inplace=True)

        self.save(trajs, 'trajs_poles')

        return poles_idx

    def project(self, poles_idx, traj_name='trajs_poles', progress=False):
        """
        """

        log.info('Compute projection along specified segments')
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
