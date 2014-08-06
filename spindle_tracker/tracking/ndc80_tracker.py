import itertools
import logging
log = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import scipy as sp

from sktracker.trajectories import Trajectories
from sktracker.tracker.solver import ByFrameSolver
from sktracker.tracker.solver import GapCloseSolver
from sktracker.utils import print_progress

from ..tracking import Tracker


class Ndc80Tracker(Tracker):

    MINIMUM_METADATA = ['SizeX', 'SizeY', 'SizeZ',
                        'PhysicalSizeX', 'PhysicalSizeY',
                        'TimeIncrement']
    ANNOTATIONS = {'state': (0, [0, 1, 2], None),
                   'anaphase_start': (-1, None, float)}

    def __init__(self, *args, **kwargs):
        """
        """

        super().__init__(*args, **kwargs)

    def track(self, trajs, n=1, progress=False, erase=False):
        """
        """

        if hasattr(self, 'trajs_gp') and getattr(self, 'trajs_gp') is not None and not erase:
            log.info('Tracking already done.')
            return

        log.info('Run tracking')

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

        trajs = solver.track(progress_bar=progress)
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

        self.remove_small_segments(trajs, n=n)
        self.save(trajs, 'trajs_gp')

    def remove_small_segments(self, trajs, n=1):
        """Remove segments smaller or equal than "n"
        """

        to_remove = filter(lambda x: len(x[1]) <= n, trajs.segment_idxs.items())
        to_remove_idx = [x[1][0] for x in to_remove]
        to_remove_idx = list(itertools.chain(*to_remove_idx))
        trajs.remove_segments(to_remove_idx)

        log.info("{} small segments has been deleted.".format(len(to_remove_idx)))

    def interp(self, traj_name='trajs_gp', erase=False):
        """
        """

        if hasattr(self, 'trajs_interp') and getattr(self, 'trajs_interp') is not None and not erase:
            log.info('Interpolation already done.')
            return

        log.info('Interpolate trajectories')
        trajs = getattr(self, traj_name).copy()

        trajs = Trajectories(trajs.time_interpolate(coords=['x', 'y', 'z', 'I', 'w']))
        trajs.drop(list(filter(lambda x: x.startswith('v_') or x.startswith('a_'), trajs.columns)),
                   axis=1, inplace=True)

        self.save(trajs, 'trajs_interp')

    def get_spb(self, traj_name='trajs_interp', progress=False, erase=False):
        """Find pairs of segments on which the mean distance on common timepoints is the bigger.
        """

        if hasattr(self, 'trajs_poles') and getattr(self, 'trajs_poles') is not None and not erase:
            trajs = getattr(self, 'trajs_poles')
            poleA = trajs[(trajs.id == 'pole') & (trajs.side == 'A')]
            poleA_id = poleA.index.get_level_values('label')[0]
            poleB = trajs[(trajs.id == 'pole') & (trajs.side == 'B')]
            poleB_id = poleB.index.get_level_values('label')[0]

            log.info("Finding poles already done")
            return (poleA_id, poleB_id)

        log.info("Finding poles")

        trajs = Trajectories(getattr(self, traj_name)).copy()

        all_d = []
        ids = []
        segs_combination = itertools.combinations(trajs.iter_segments, 2)
        n = len(trajs.segment_idxs)
        n = sp.misc.factorial(n) / (sp.misc.factorial(n - 2) * sp.misc.factorial(2))
        for i, ((id1, seg1), (id2, seg2)) in enumerate(segs_combination):

            if progress:
                print_progress(i * 100 / n)

            seg1 = seg1.loc[:, ['x', 'y', 't']].reset_index(drop=True)
            seg2 = seg2.loc[:, ['x', 'y', 't']].reset_index(drop=True)

            merged = pd.merge(seg1.reset_index(), seg2.reset_index(), on='t')

            if not merged.empty:
                p1 = merged.loc[:, ['x_x', 'y_x']].values
                p2 = merged.loc[:, ['x_y', 'y_y']].values
                vec = p1 - p2
                d = np.sqrt(vec[:, 0]**2 + vec[:, 1]**2)
                d_mean = d.mean()

                length_ratio = (seg1.shape[0] * seg2.shape[0]) / (seg1.shape[0] + seg2.shape[0])
                score = d_mean * length_ratio

                all_d.append(score)
                ids.append((id1, id2))

        if progress:
            print_progress(-1)

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

    def project(self, poles_idx, traj_name='trajs_poles', progress=False, erase=False):
        """
        """

        if hasattr(self, 'trajs_project') and getattr(self, 'trajs_project') is not None and not erase:
            log.info('Projection already done.')
            return

        log.info('Compute projection along specified segments')
        trajs = getattr(self, traj_name).copy()
        trajs = trajs.project(poles_idx,
                              keep_first_time=False,
                              reference=None,
                              inplace=False,
                              progress=progress)

        self.save(trajs, 'trajs_project')

    def guess_n_kts(self, traj_name='trajs_project'):
        """
        """
        trajs = getattr(self, traj_name)

        def guess_n_kts(peaks):
            peaks['n_kts'] = np.round(peaks['I'] * 8 / peaks['I'].sum()).values
            return peaks

        trajs = trajs.groupby(level='t_stamp').apply(guess_n_kts)

        setattr(self, traj_name, Trajectories(trajs))

    def kymo(self, var_name='trajs', marker='o', ls='-', ax=None):
        """
        """

        import matplotlib.pyplot as plt

        trajs = Trajectories(getattr(self, var_name))

        if ax is None:
            fig, ax = plt.subplots(nrows=1)
        else:
            fig = ax.get_figure()
        ax = trajs.show(xaxis='t', yaxis='x_proj',
                        groupby_args={'level': "label"},
                        ax=ax, ls=ls, marker=marker)

        return fig
