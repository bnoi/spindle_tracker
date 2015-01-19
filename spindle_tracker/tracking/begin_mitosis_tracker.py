import gc
import logging

log = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import scipy
from skimage import measure

from sktracker.trajectories import Trajectories
from sktracker.tracker.solver import ByFrameSolver
from sktracker.io import TiffFile

from ..tracking import Tracker


class BeginMitosisTracker(Tracker):

    ANNOTATIONS = {'start_mitosis': (-1, None, float),
                   'state': (0, [0, 1, 2], None)}

    def __init__(self, *args, **kwargs):
        """
        """

        super().__init__(*args, **kwargs)

        if hasattr(self, 'line_size'):
            self.line_size = pd.Series(self.line_size)

    def track_poles(self, force=False):
        """
        """

        if force or not hasattr(self, 'poles'):
            poles = self.get_peaks_from_trackmate()
            poles = poles.groupby(level='t_stamp').filter(lambda x: len(x) == 2)
            poles = Trajectories(poles)

            solver = ByFrameSolver.for_brownian_motion(poles,
                                                       max_speed=1e10, coords=['x', 'y'])
            poles = solver.track(progress_bar=True)

            poles = poles.project([0, 1], keep_first_time=False,
                                  reference=None, inplace=False, progress=True)

            self.save(poles, 'poles')

    def get_line_profiles(self, lw=0.7, force=False):
        """
        """

        if force or not hasattr(self, 'line_profiles'):

            # Get image
            tf = self.get_tif()
            im = tf.asarray()
            tf.close()
            md = self.metadata

            # Z projection
            id_z = md['DimensionOrder'].index('Z')
            im = im.max(axis=id_z)

            # Get GFP channel
            id_c =  im.shape.index(2)
            id_ndc80 = md['Channels'].index('GFP')
            gfp_im = im.take(id_ndc80, axis=id_c)
            gfp_im = gfp_im / np.median(gfp_im)

            del im
            gc.collect()

            gfp_im = (gfp_im - gfp_im.min()) / (gfp_im.max() - gfp_im.min())

            lw_pixel = lw / md['PhysicalSizeX']

            line_profiles = {}
            line_size = {}

            for t_stamp, p in self.poles.groupby(level='t_stamp'):
                a = gfp_im[t_stamp]

                scaled_p = p.copy()
                scaled_p.loc[:, ['x', 'y', 'w']] /= md['PhysicalSizeX']

                p1 = scaled_p.iloc[0][['y', 'x']]
                p2 = scaled_p.iloc[1][['y', 'x']]
                lp = measure.profile_line(a, p1, p2, linewidth=lw_pixel)

                line_profiles[t_stamp] = lp
                line_size[t_stamp] = scipy.spatial.distance.cdist(np.atleast_2d(p1.values), np.atleast_2d(p2.values))[0, 0]
                line_size[t_stamp] *= self.metadata['PhysicalSizeX']

            del gfp_im
            del tf

            gc.collect()

            line_profiles = pd.DataFrame.from_dict(line_profiles, orient='index')
            line_size = pd.Series(line_size)
            self.save(line_profiles, 'line_profiles')
            self.save(line_size, 'line_size')
