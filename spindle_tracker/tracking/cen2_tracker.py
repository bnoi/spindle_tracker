import logging

import numpy as np
import scipy as sp
import pandas as pd
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hier

log = logging.getLogger(__name__)

from sktracker.tracker.solver import ByFrameSolver
from sktracker.utils import print_progress
from sktracker.trajectories import Trajectories

from ..tracking import Tracker


class Cen2Tracker(Tracker):

    MINIMUM_METADATA = ['SizeX', 'SizeY', 'SizeZ',
                        'PhysicalSizeX', 'PhysicalSizeY',
                        'TimeIncrement']
    ANNOTATIONS = {'state': (0, [0, 1, 2], None),
                   'anaphase_start': (-1, None, float)}

    def __init__(self, *args, **kwargs):
        """
        """

        super().__init__(*args, **kwargs)

    @property
    def times(self):
        return self.peaks_real['t'].unique().astype('float')

    @property
    def times_interpolated(self):
        return self.peaks_real_interpolated['t'].unique().astype('float')

    @property
    def frames(self):
        return self.peaks_z['t'].unique().astype('float')

    @property
    def anaphase(self):
        try:
            return self.annotations['anaphase']
        except:
            return None

    @property
    def index_anaphase(self):
        if self.anaphase and self.anaphase != -1:
            return self.get_closest_time(self.anaphase, interpolated=False)
        else:
            return None

    @property
    def index_anaphase_interpolated(self):
        if self.anaphase:
            return self.get_closest_time(self.anaphase, interpolated=True)
        else:
            return None

    def get_closest_time(self, t, interpolated=False):
        """
        """
        if interpolated:
            peaks = self.peaks_real_interpolated
        else:
            peaks = self.peaks_real

        i = int(np.argmin(np.abs(self.anaphase - peaks['t'].unique())))
        return peaks.index.get_level_values('t_stamp').unique()[i]

    def get_time_from_stamp(self, vec_t_stamp, interpolated=False):
        """
        """
        if interpolated:
            peaks = self.peaks_real_interpolated
        else:
            peaks = self.peaks_real

        return peaks.loc[vec_t_stamp, 't'].unique()

    """
    Tracking methods
    """

    def find_z(self, treshold=0.1, erase=False, use_trackmate=True):
        """
        Find peaks with same x and y coordinate (with a cluster algorithm).
        Keep the peak with biggest intensity and add z coordinates
        according to his position in the z-stack.

        Parameters
        ----------
        treshold: float
            Treshold value for the clustering algorithm. It corresponds to
            the minimum distance to say if two points are close to each
            other.
        """
        if hasattr(self, "peaks_z") and isinstance(self.peaks_z, pd.DataFrame) and not erase:
            return self.peaks_z

        if use_trackmate:
            self.get_peaks_from_trackmate()

        if hasattr(self, 'raw_trackmate') and use_trackmate:
            log.info("Use TrackMate to import detected peaks")
            peaks = self.raw_trackmate.copy()
        else:
            peaks = self.raw.copy()

        z_position = self.metadata['DimensionOrder'].index('Z')
        z_in_raw = peaks['z'].unique().shape[0]
        if self.metadata['Shape'][z_position] == 1 or z_in_raw == 1:
            log.info('No Z detected, pass Z projection clustering.')
            self.peaks_z = peaks
            return

        log.info("*** Running find_z()")

        bads = []
        clusters_count = []

        for t, pos in peaks.groupby('t'):
            if pos.shape[0] > 1:
                dist_mat = dist.squareform(dist.pdist(pos[['x', 'y']]))
                link_mat = hier.linkage(dist_mat)
                clusters = hier.fcluster(
                    link_mat, treshold, criterion='distance')
                pos['clusters'] = clusters
                for cluster, p in pos.groupby('clusters'):
                    bads_peak = p.sort(columns='I').iloc[:-1].index.values
                    bads.extend(bads_peak)
                    clusters_count.append(len(p))
            else:
                for p in pos.iterrows():
                    clusters_count.append(0)

        peaks_z = peaks.drop(bads)
        peaks_z['clusters_count'] = clusters_count
        self.save(peaks_z, 'peaks_z')

        log.info("*** End")

    def track(self,
              v_max,
              num_kept,
              max_radius,
              v_max_spb=np.inf,
              coords=['x', 'y'],
              erase=False,
              reference=None,
              keep_first_time=False):
        """
        Set real coordinates. This process contains severals steps:
            1. Remove weak peaks
            2. Remove timepoints with less than 4 peaks
            3. Label peaks (spb or Kt and side A or side B)
            4. Remove outliers peaks which exceed a given speed
            5. Project x, y coordinates to single axis (1,0)
               (x_proj in /peaks_real)
        """

        if hasattr(self, 'peaks_real') and isinstance(self.peaks_real, pd.DataFrame) and not erase:
            return self.peaks_real

        self.peaks_real = None
        peaks_real = self.peaks_z.copy()
        self.save(peaks_real, 'peaks_real')

        self._remove_weakers(num_kept=num_kept, max_radius=max_radius)
        self._remove_uncomplete_timepoints(num_kept=num_kept)

        if not self.peaks_real.empty:
            self._label_peaks(coords=coords)
            self._label_peaks_side(v_max=v_max_spb, coords=coords)
            if v_max:
                self._remove_outliers(v_max=v_max)

            self.project(coords=coords,
                         reference=reference,
                         keep_first_time=keep_first_time)

            self._interpolate()

            self.peaks_real.sort_index(inplace=True)
        else:
            log.error("peaks_real is empty")

    def _remove_weakers(self,
                        num_kept=4,
                        max_radius=0.3):
        """
        """

        log.info("*** Running _remove_weakers()")

        peaks = self.peaks_real
        bads = []

        n_bads_intensity = 0
        n_bads_radius = 0

        for t, pos in peaks.groupby('t'):

            # Remove peaks with radius greater than treshold
            bads_radius_index = pos[pos['w'] > max_radius].index.values
            bads.extend(bads_radius_index)
            n_bads_radius += len(bads_radius_index)

            # Select n peaks by intensity or quality
            if pos.shape[0] >= num_kept:
                if 'q' in pos.columns:
                    bads_intensity_index = pos.sort('q').iloc[:-num_kept].index.values
                else:
                    bads_intensity_index = pos.sort('I').iloc[:-num_kept].index.values
                bads.extend(bads_intensity_index)
                n_bads_intensity += len(bads_intensity_index)

        bads = list(set(bads))

        mess = '{} peaks removed for weak intensity/quality.'
        log.info(mess.format(n_bads_intensity))
        mess = '{} peaks removed for too large radius'
        log.info(mess.format(n_bads_radius))
        log.info('Total removed: {} / {} peaks'.format(len(bads), len(peaks)))

        self.peaks_real = peaks.drop(bads)

        log.info("*** End")

    def _remove_uncomplete_timepoints(self, num_kept=4):
        """
        """

        log.info("*** Running _remove_uncomplete_timepoints()")

        peaks = self.peaks_real.copy()
        bads = []

        removed_t = 0
        num_removed = []
        for t, pos in peaks.groupby('t'):
            if len(pos) < num_kept:
                bads.extend(pos.index)
                removed_t += 1
            num_removed.append(len(pos))

        bads = list(set(bads))
        self.peaks_real = peaks.drop(bads)

        n, i = np.histogram(num_removed, bins=[0, 1, 2, 3, 4, 5])
        log.info('Number of peaks by timepoints : {}'.format(dict(zip(i, n))))

        n_unique_peaks = len(np.unique(peaks['t'].values))
        log.info('{} / {} uncomplete timepoints removed'.format(removed_t, n_unique_peaks))

        log.info("*** End")

    def _label_peaks(self, coords=['x', 'y']):
        """
        Labels SPB and Kt peaks

        Return
        ------
            self.peaks_real: DataFrame
        """

        log.info("*** Running _label_peaks()")

        peaks = self.peaks_real

        peaks.reset_index('label', inplace=True)
        del peaks['label']

        peaks['main_label'] = 'kt'

        n = self.times.size
        for i, (t_stamp, p) in enumerate(peaks.groupby(level='t_stamp')):

            if self.verbose:
                print_progress(i * 100 / n)

            d = dist.squareform(dist.pdist(p[coords]))
            i, j = np.unravel_index(d.argmax(), d.shape)

            peaks['main_label'].loc[t_stamp].iloc[[i, j]] = 'spb'

        if self.verbose:
            print_progress(-1)

        peaks.set_index(['main_label'], append=True, inplace=True)

        self.peaks_real = peaks

        log.info("*** End")

    def _label_peaks_side(self, v_max=np.inf, coords=['x', 'y']):
        """
        Label side for every couple peaks. Side is 'A' or 'B'.

        Return
        ------
            self.peaks_real: DataFrame
        """

        log.info("*** Running _label_peaks_side()")

        peaks = self.peaks_real
        if 'side' in peaks.columns:
            peaks = peaks.reset_index('side').drop('side', axis=1)

        # Split peaks
        peaks = peaks.swaplevel("main_label", "t_stamp")
        spbs = peaks.loc['spb']
        kts = peaks.loc['kt']
        kts['side'] = np.nan
        spbs['side'] = np.nan

        # Track SPBs
        spbs['label'] = range(spbs.shape[0])
        spbs.set_index('label', append=True, inplace=True)

        solver = ByFrameSolver.for_brownian_motion(spbs, max_speed=v_max, coords=coords)
        spbs = solver.track(progress_bar=True)

        spbs.reset_index(level='label', inplace=True)
        spbs.loc[:, 'side'][spbs.label == 0] = 'A'
        spbs.loc[:, 'side'][spbs.label == 1] = 'B'

        # Prepare to merge
        spbs['main_label'] = 'spb'
        kts['main_label'] = 'kt'

        # Do the merge
        peaks = pd.concat([kts, spbs]).sort_index()

        def label_kts(p):

            spb1 = p[(p['main_label'] == 'spb') & (p['side'] == 'A')]

            kt1 = p[(p['main_label'] == 'kt')].iloc[0]
            kt2 = p[(p['main_label'] == 'kt')].iloc[1]

            kt1_dist = np.linalg.norm(spb1[coords] - kt1[coords])
            kt2_dist = np.linalg.norm(spb1[coords] - kt2[coords])

            if kt1_dist < kt2_dist:
                p.loc[:, 'side'][(p['main_label'] == 'kt')] = ['A', 'B']
                p.loc[:, 'label'][(p['main_label'] == 'kt')] = [2, 3]
            else:
                p.loc[:, 'side'][(p['main_label'] == 'kt')] = ['B', 'A']
                p.loc[:, 'label'][(p['main_label'] == 'kt')] = [3, 2]

            return p

        peaks = peaks.groupby(level='t_stamp').apply(label_kts)
        peaks.set_index(['main_label', 'side'], append=True, inplace=True)

        self.peaks_real = peaks

        log.info("*** End")

    def _remove_outliers(self, v_max):
        """
        Remove outliers peaks which overtake a speed given as a parameter.
        self.peaks_real is overriden

        Parameters
        ----------
        v_max: float
            maximum speed for one peak (in µm/s)
        """

        log.info("*** Running _remove_outliers()")

        def get_speed(pos):
            x = pos['x']
            y = pos['y']
            t = pos['t']
            dr = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
            dt = np.diff(t)
            v = dr / dt
            return v, t

        t_stamp_to_remove = []

        for (main_label, side), peaks in self.peaks_real.groupby(level=('main_label', 'side')):

            out = False
            while not out:

                v, t = get_speed(peaks)
                v_above_treshold = np.argwhere(v > v_max)

                if len(v_above_treshold) > 0:
                    idx = v_above_treshold[0]
                    t_stamp_idx = peaks.iloc[idx + 1].index.get_level_values('t_stamp')[0]
                    peaks = peaks.drop(t_stamp_idx, level='t_stamp')
                    t_stamp_to_remove.append(t_stamp_idx)
                else:
                    out = True

        n = np.unique(t_stamp_to_remove).size
        tot = self.peaks_real['t'].unique().size
        self.peaks_real.drop(t_stamp_to_remove, level='t_stamp', inplace=True)
        log.info("{} / {} timepoints removed because of outliers".format(n, tot))

        log.info("*** End")

        return self.peaks_real

    def project(self,
                coords=['x', 'y'],
                reference=None,
                keep_first_time=False):
        """
        Project peaks from 2D to 1D taking SPB - SPB for the main axis.

        Returns
        -------
            self.peaks_real: DataFrame
        """

        log.info("*** Running project()")

        progress = True
        peaks = self.peaks_real.copy()

        peaks = peaks.reset_index(level=["main_label", "side"])
        peaks['label'] = np.nan

        peaks.loc[:, 'label'][(peaks['main_label'] == 'kt') & (peaks['side'] == 'A')] = 0
        peaks.loc[:, 'label'][(peaks['main_label'] == 'kt') & (peaks['side'] == 'B')] = 1
        peaks.loc[:, 'label'][(peaks['main_label'] == 'spb') & (peaks['side'] == 'A')] = 2
        peaks.loc[:, 'label'][(peaks['main_label'] == 'spb') & (peaks['side'] == 'B')] = 3

        peaks = peaks.set_index('label', append=True)

        peaks = Trajectories(peaks)

        peaks = peaks.project([2, 3],
                              keep_first_time=False,
                              reference=None,
                              inplace=False,
                              progress=progress)

        peaks = peaks.reset_index(level='label').set_index(['main_label', 'side'], append=True)
        self.peaks_real = peaks

        log.info("*** End")

    def _interpolate(self, dt=1, kind='linear'):
        """
        Interpolate data for x, y, z and x_proj. Then create new
        /peaks_real_interpolated with new times.

        TODO: use sktracker.trajectories.Trajectories.interpolate()
        """

        log.info("*** Running _interpolate()")

        peaks = self.peaks_real
        peaks_interp = pd.DataFrame([])

        for (label, side), p in peaks.groupby(level=['main_label', 'side']):
            new_p = pd.DataFrame(np.arange(np.round(p['t'].min()) + 1, p['t'].max(), dt),
                                 columns=['t'])
            new_p['main_label'] = label
            new_p['side'] = side

            for dim in p.columns:
                if dim not in ['id', 't']:
                    fc = sp.interpolate.interp1d(p['t'], p[dim], kind=kind)
                    new_p[dim] = fc(new_p['t'])

            new_p['t_stamp'] = np.arange(0, new_p.shape[0])

            peaks_interp = peaks_interp.append(new_p)

        peaks_interp.set_index(['t_stamp', 'main_label', 'side'], inplace=True)
        peaks_interp.sort(inplace=True)

        self.save(peaks_interp, 'peaks_real_interpolated')

        log.info("*** End")

    """
    Plot and visu methods
    """

    def kymo(self, use_interpolate=False, time_in_minutes=False,
             mpl_params={'ls': '-', 'marker': 'o'}):
        """
        Print a kymograph of DataFrame of peaks (self.peaks_real{'x_proj']})
        """

        if self.peaks_real.empty:
            log.error("peaks_real is empty")
            return None

        if use_interpolate:
            peaks = self.peaks_real_interpolated
            times = self.times_interpolated
        else:
            peaks = self.peaks_real
            times = self.times

        colors = ["#A25540",  # SPB A
                  "#95BD55",  # SPB B
                  "#7F9C9A",  # Kt A
                  "#9E5EAB"]  # Kt B

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 7))
        ax = plt.subplot(111)

        drawer = ax.plot

        gps = peaks.groupby(level=['main_label', 'side']).groups
        coord = 'x_proj'

        # Draw SPB
        x = peaks.loc[gps[('spb', 'A')]][coord]
        drawer(times, x, label="SPB A", color=colors[0], **mpl_params)

        x = peaks.loc[gps[('spb', 'B')]][coord]
        drawer(times, x, label="SPB B", color=colors[1], **mpl_params)

        # Draw Kt
        x = peaks.loc[gps[('kt', 'A')]][coord]
        drawer(times, x, label="Kt A", color=colors[2], **mpl_params)

        x = peaks.loc[gps[('kt', 'B')]][coord]
        drawer(times, x, label="Kt B", color=colors[3], **mpl_params)

        # Set axis limit
        ax.set_xlim(min(times), max(times))
        m = np.abs(peaks['x_proj'].max())
        ax.set_ylim(-m, m)
        # ax.set_ylim(-2, 2)

        fontsize = 22

        if time_in_minutes:
            import matplotlib

            majorLocator = matplotlib.ticker.MultipleLocator(60)
            minorLocator = matplotlib.ticker.MultipleLocator(60)
            ax.xaxis.set_major_locator(majorLocator)
            ax.xaxis.set_minor_locator(minorLocator)

            majorFormatter = matplotlib.ticker.FuncFormatter(
                lambda x, y: "%.0f" % (x / 60.))
            ax.xaxis.set_major_formatter(majorFormatter)

            ax.set_xlabel('Time (mn)', fontsize=fontsize)

        else:
            ax.set_xlabel('Time (seconds)', fontsize=fontsize)

        ax.set_title("Kymograph like plot", fontsize=fontsize)
        ax.set_ylabel('Distance (µm)', fontsize=fontsize)

        if hasattr(self, 'analysis') and 'anaphase_start' in self.analysis.keys():
            if self.analysis['anaphase_start']:
                ax.axvline(x=self.analysis['anaphase_start'],
                           color='black',
                           alpha=1,
                           linestyle="--",
                           label='Anaphase start')

        leg = ax.legend(loc='best', fancybox=True)
        leg.get_frame().set_alpha(0.5)

        plt.grid(True)
        plt.tight_layout()

        return fig

    def kymo_figure(self):
        """
        """

        if self.peaks_real.empty:
            log.error("peaks_real is empty")
            return None

        peaks = self.peaks_real_interpolated
        times = self.times_interpolated

        colors = ["black",  # SPB A
                  "black",  # SPB B
                  "#ff2626",  # Kt A
                  "#ff2626"]  # Kt B

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(3.6*5, 5))
        ax = plt.subplot(111)

        drawer = ax.plot

        gps = peaks.groupby(level=['main_label', 'side']).groups
        coord = 'x_proj'

        # Draw SPB
        x = peaks.loc[gps[('spb', 'A')]][coord]
        drawer(times, x, label="SPB A", color=colors[0], lw=3)

        x = peaks.loc[gps[('spb', 'B')]][coord]
        drawer(times, x, label="SPB B", color=colors[1], lw=3)

        # Draw Kt
        x = peaks.loc[gps[('kt', 'A')]][coord]
        drawer(times, x, label="Kt A", color=colors[2], lw=3)

        x = peaks.loc[gps[('kt', 'B')]][coord]
        drawer(times, x, label="Kt B", color=colors[3], lw=3)

        # Set axis limit
        ax.set_xlim(min(times), max(times))
        ax.set_ylim(-6, 6)

        import matplotlib

        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        majorLocator = matplotlib.ticker.MultipleLocator(60*10)
        ax.xaxis.set_major_locator(majorLocator)
        ax.minorticks_off()

        majorFormatter = matplotlib.ticker.FuncFormatter(lambda x, y: "")
        ax.xaxis.set_major_formatter(majorFormatter)
        majorFormatter = matplotlib.ticker.FuncFormatter(lambda x, y: "")
        ax.yaxis.set_major_formatter(majorFormatter)

        majorLocator = matplotlib.ticker.MultipleLocator(4)
        ax.yaxis.set_major_locator(majorLocator)

        for i in ax.spines.values():
            i.set_linewidth(2)
            i.set_color('black')

        ax.grid(b=True, which='major', color='#555555', linestyle='-', alpha=1, lw=1)

        return fig
