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
            try:
                id_ndc80 = md['Channels'].index('GFP')
            except:
                id_ndc80 = 0
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

    def get_figure(self, figsize=(13, 8)):
        """
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        fig, ax = plt.subplots(figsize=figsize)

        pole_1 = self.poles.loc[pd.IndexSlice[:, 0], ]
        pole_2 = self.poles.loc[pd.IndexSlice[:, 1], ]
        ax.plot(pole_1['t'], pole_1 ['x_proj'], c='black', marker='o')
        ax.plot(pole_2['t'], pole_2['x_proj'], c='black', marker='o')

        precision = 1000
        linewidth = 6
        alpha = 1
        norm = plt.Normalize(0.0, 1.0)
        cmap = plt.get_cmap('Reds')
        #cmap.set_gamma(2)

        for t_stamp, p in self.poles.groupby(level='t_stamp'):
            lp = self.line_profiles.loc[t_stamp]

            p1 = p.iloc[0][['x_proj']].values[0]
            p2 = p.iloc[1][['x_proj']].values[0]

            x = np.repeat(p['t'].unique()[0], precision)
            y = np.linspace(p1, p2, num=precision)

            # Get color vector according to line profile
            lp = lp.dropna().values
            lp = (lp - lp.min()) / (lp.max() - lp.min())
            x_lp = np.arange(0, len(lp))
            new_x_lp = np.linspace(0, len(lp) - 1, precision)
            z = np.interp(new_x_lp, x_lp, lp)

            # Make segments
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
            ax.add_collection(lc)

        return fig

    def get_figure_publi(self, figsize, tmin, tmax):
        """
        """
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        fig, ax = plt.subplots(figsize=figsize)

        poles = self.poles[self.poles['t'] < tmax]
        poles = poles[self.poles['t'] > tmin]
        poles['t'] = poles['t'] - self.poles.loc[self.annotations['start_mitosis'], 't'].iloc[0]
        poles['t'] = poles['t'] / 60

        pole_1 = poles.loc[pd.IndexSlice[:, 0], ]
        pole_2 = poles.loc[pd.IndexSlice[:, 1], ]

        times = pole_1['t'].values
        ax.plot(times, pole_1 ['x_proj'], color='#000000', marker='o')
        ax.plot(times, pole_2['x_proj'], color='#000000', marker='o')

        precision = 1000
        linewidth = 6
        alpha = 1
        norm = plt.Normalize(0.0, 1.0)
        cmap = plt.get_cmap('Reds')

        for t_stamp, p in poles.groupby(level='t_stamp'):
            lp = self.line_profiles.loc[t_stamp]

            p1 = p.iloc[0][['x_proj']].values[0]
            p2 = p.iloc[1][['x_proj']].values[0]

            x = np.repeat(p['t'].unique()[0], precision)
            y = np.linspace(p1, p2, num=precision)

            # Get color vector according to line profile
            lp = lp.dropna().values
            lp = (lp - lp.min()) / (lp.max() - lp.min())
            x_lp = np.arange(0, len(lp))
            new_x_lp = np.linspace(0, len(lp) - 1, precision)
            z = np.interp(new_x_lp, x_lp, lp)

            # Make segments
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
            ax.add_collection(lc)

        ax.set_xticks(np.arange(-8, 8, 2))
        ax.set_xlim(times[0], times[-1])
        ax.set_yticks(np.arange(-0.8, 0.8, 0.4))

        nullform = matplotlib.ticker.FuncFormatter(lambda x, y: "")
        ax.xaxis.set_major_formatter(nullform)
        ax.yaxis.set_major_formatter(nullform)

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        for i in ax.spines.values():
            i.set_linewidth(4)
            i.set_color('black')

        ax.grid(b=True, which='major', color='#000000', linestyle='-', alpha=0.4, lw=2)

        plt.tight_layout()
        return fig
