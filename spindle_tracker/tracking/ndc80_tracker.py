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

    def track(self, trajs,
              max_speed,
              death_birth_factor,
              maximum_gap,
              gap_close_factor,
              min_segment_size,
              coords=['x', 'y', 'z'],
              progress=False,
              erase=False):
        """
        """

        if hasattr(self, 'trajs_gap_close') and getattr(self, 'trajs_gap_close') is not None and \
           not erase:
            log.info('Tracking already done.')
            return

        self.save(trajs.copy(), 'trajs')
        log.info('Run tracking')

        parameters_brownian = {'max_speed': max_speed,
                               'coords': coords,
                               'penalty': death_birth_factor}

        solver = ByFrameSolver.for_brownian_motion(trajs, **parameters_brownian)
        trajs = solver.track(progress_bar=progress)

        self.save(trajs.copy(), 'trajs_by_frame')

        trajs = self.trajs_by_frame.copy()

        parameters_gap_close = {'max_speed': max_speed,
                                'maximum_gap': maximum_gap,
                                'use_t_stamp': False,
                                'link_percentile': gap_close_factor,
                                'coords': coords}

        gc_solver = GapCloseSolver.for_brownian_motion(trajs, **parameters_gap_close)
        trajs = gc_solver.track(progress_bar=progress)

        self.remove_small_segments(trajs, n=min_segment_size)
        self.save(trajs, 'trajs_gap_close')

    def remove_small_segments(self, trajs, n=1):
        """Remove segments smaller or equal than "n"
        """

        to_remove = filter(lambda x: len(x[1]) <= n, trajs.segment_idxs.items())
        to_remove_idx = [x[1][0] for x in to_remove]
        to_remove_idx = list(itertools.chain(*to_remove_idx))
        trajs.remove_segments(to_remove_idx)

        log.info("{} small segments has been deleted.".format(len(to_remove_idx)))

    def interpolate(self, traj_name='trajs_gap_close', erase=False):
        """
        """

        if hasattr(self, 'trajs_interp') and getattr(self, 'trajs_interp') is not None \
           and not erase:
            log.info('Interpolation already done.')
            return

        log.info('Interpolate trajectories')
        trajs = getattr(self, traj_name).copy()

        trajs = Trajectories(trajs.time_interpolate(coords=['x', 'y', 'z', 'I', 'w']))
        trajs.drop(list(filter(lambda x: x.startswith('v_') or x.startswith('a_'), trajs.columns)),
                   axis=1, inplace=True)

        self.save(trajs, 'trajs_interp')

    def find_poles(self, traj_name='trajs_interp', progress=False, erase=False):
        """Find pairs of segments on which the mean distance on common timepoints is the bigger.
        """

        if hasattr(self, 'annotations') and 'poles_index' in self.annotations.keys() and not erase:
            self.poles_index = self.annotations['poles_index']
            # log.info("Poles already found : {}".format(self.poles_index))
            return self.poles_index

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
        poles_index = ids[max_id]

        # Label trajs correctly
        trajs['id'] = None
        trajs['side'] = None
        trajs = trajs.swaplevel("label", "t_stamp")
        trajs.sort_index(inplace=True)

        for i, l in zip(poles_index, ['A', 'B']):
            trajs.loc[i, 'id'] = 'pole'
            trajs.loc[i, 'side'] = l

        trajs = trajs.swaplevel("label", "t_stamp")
        trajs.sort_index(inplace=True)

        self.save(trajs, 'trajs_poles')

        self.poles_index = poles_index
        self.annotations['poles_index'] = self.poles_index
        log.info("Poles are {}".format(self.poles_index))

        return np.sort(poles_index)

    def find_kts(self):
        """Kts are all spots excluded both labeled as poles.
        """

        poles = self.find_poles()
        return np.delete(self.trajs.labels, poles)

    def project(self, traj_name='trajs_poles', progress=False, erase=False):
        """
        """

        if hasattr(self, 'trajs_project') and getattr(self, 'trajs_project') is not None and \
           not erase:
            log.info('Projection already done.')
            return

        if not hasattr(self, 'poles_index'):
            log.error("self.poles_index is missing, please run find_poles() first.")
            return

        log.info('Compute projection along specified segments')
        trajs = getattr(self, traj_name).copy()
        trajs = trajs.project(self.poles_index,
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

    def kymo(self, var_name, marker='o', ls='-', ax=None):
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
