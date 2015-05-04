import logging
import os
import itertools

import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hier

log = logging.getLogger(__name__)

from sktracker.tracker.solver import ByFrameSolver
from sktracker.utils import progress_apply
from sktracker.trajectories import Trajectories
from sktracker.io.trackmate import trackmate_peak_import
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
    def times_metaphase(self):
        return self.get_peaks_metaphase()['t'].unique().astype('float')

    @property
    def times_interpolated_metaphase(self):
        return self.get_peaks_interpolated_metaphase()['t'].unique().astype('float')

    @property
    def times_anaphase(self):
        return self.get_peaks_anaphase()['t'].unique().astype('float')

    @property
    def times_interpolated_anaphase(self):
        return self.get_peaks_interpolated_anaphase()['t'].unique().astype('float')

    @property
    def frames(self):
        return self.peaks_z['t'].unique().astype('float')

    @property
    def anaphase(self):
        try:
            return self.annotations['anaphase_start']
        except:
            return None

    @property
    def index_anaphase(self):
        if self.anaphase:
            if self.anaphase != -1:
                return self.get_closest_time(self.anaphase, interpolated=False)
            else:
                return -1
        else:
            return None

    @property
    def index_anaphase_interpolated(self):
        if self.anaphase:
            if self.anaphase != -1:
                return self.get_closest_time(self.anaphase, interpolated=True)
            else:
                return -1
        else:
            return None

    def get_closest_time(self, t, interpolated=False):
        """
        """
        if interpolated:
            peaks = self.peaks_real_interpolated
        else:
            peaks = self.peaks_real

        i = int(np.argmin(np.abs(t - peaks['t'].unique())))
        return peaks.index.get_level_values('t_stamp').unique()[i]


    def get_time_from_stamp(self, vec_t_stamp, interpolated=False):
        """
        """
        if interpolated:
            peaks = self.peaks_real_interpolated
        else:
            peaks = self.peaks_real

        return peaks.loc[vec_t_stamp, 't'].unique()

    def get_peaks_interpolated_metaphase(self):
        """
        """
        return self.get_peaks(step='metaphase', interpolated=True)

    def get_peaks_metaphase(self):
        """
        """
        return self.get_peaks(step='metaphase', interpolated=False)

    def get_peaks_interpolated_anaphase(self):
        """
        """
        return self.get_peaks(step='anaphase', interpolated=True)

    def get_peaks_anaphase(self):
        """
        """
        return self.get_peaks(step='anaphase', interpolated=False)

    def get_peaks(self, step, interpolated=False):
        """step can be 'metaphase' or 'anaphase'
        """

        if interpolated:
            index_anaphase = self.index_anaphase_interpolated
        else:
            index_anaphase = self.index_anaphase

        if index_anaphase is not None and index_anaphase > 0:
            if interpolated and step == 'metaphase':
                peaks = self.peaks_real_interpolated.loc[:index_anaphase]
            elif interpolated and step == 'anaphase':
                peaks = self.peaks_real_interpolated.loc[index_anaphase:]
            elif not interpolated and step == 'metaphase':
                peaks = self.peaks_real.loc[:index_anaphase]
            elif not interpolated and step == 'anaphase':
                peaks = self.peaks_real.loc[index_anaphase:]
            else:
                raise Exception('Error in get_peaks()')

        else:
            if interpolated and step == 'metaphase':
                peaks = self.peaks_real_interpolated
            elif not interpolated and step == 'metaphase':
                peaks = self.peaks_real_interpolated
            else:
                raise Exception('Error in get_peaks()')

        return peaks

    """
    Tracking methods
    """

    def load_peaks(self, use_trackmate=True):
        """
        """

        if use_trackmate:
            self.get_peaks_from_trackmate()

        if hasattr(self, 'raw_trackmate') and use_trackmate:
            log.info("Use TrackMate to import detected peaks")
            peaks = self.raw_trackmate.copy()
        else:
            peaks = self.raw.copy()

        self.save(peaks, 'peaks')

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

        z_position = self.metadata['DimensionOrder'].index('Z')
        z_in_raw = self.peaks['z'].unique().shape[0]
        if self.metadata['Shape'][z_position] == 1 or z_in_raw == 1:
            log.info('No Z detected, pass Z projection clustering.')
            self.peaks_z = self.peaks.copy()
            return

        log.info("*** Running find_z()")

        bads = []
        clusters_count = []

        peaks = self.peaks

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

        if hasattr(self, 'peaks_real') and isinstance(self.peaks_real, pd.DataFrame) and \
           not erase:
            return self.peaks_real

        if hasattr(self, 'peaks_z'):
            log.info("Using 'peaks_z'")
            peaks_real = self.peaks_z.copy()
        else:
            peaks_real = self.peaks.copy()

        self.save(peaks_real, 'peaks_real')

        self._remove_weakers(num_kept=num_kept, max_radius=max_radius)
        self.remove_uncomplete_timepoints(num_kept=num_kept)

        if not self.peaks_real.empty:
            self._label_peaks(coords=coords)
            self._label_peaks_side(v_max=v_max_spb, coords=coords)
            if v_max:
                self._remove_outliers(v_max=v_max)

            self.project(coords=coords,
                         reference=reference,
                         keep_first_time=keep_first_time)

            self.interpolate()

            self.peaks_real.sort_index(inplace=True)

            self.unswitch_to_label(self.peaks_real)
            self.unswitch_to_label(self.peaks_real_interpolated)

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

    def remove_uncomplete_timepoints(self, num_kept=4):
        """
        """

        log.info("*** Running remove_uncomplete_timepoints()")

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

        if 'label' in peaks.index.names:
            peaks.reset_index('label', inplace=True)

        peaks['main_label'] = 'kt'

        def get_spb(x):
            d = dist.squareform(dist.pdist(x[coords]))
            idxs = np.unravel_index(d.argmax(), d.shape)
            x.loc[:, 'main_label'].iloc[list(idxs)] = "spb"
            return x

        gp = peaks.groupby(level='t_stamp')
        peaks = progress_apply(gp, get_spb)

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
        if 'side' in peaks.index.names:
            peaks.reset_index('side', inplace=True)
        if 'side' in peaks.columns:
            peaks.drop('side', axis=1, inplace=True)

        # Split peaks
        idx = pd.IndexSlice
        peaks.sortlevel(inplace=True)
        spbs = peaks.loc[idx[:, 'spb'], :]
        kts = peaks.loc[idx[:, 'kt'], :]

        kts.loc[:, 'side'] = np.nan
        spbs.loc[:, 'side'] = np.nan

        # Track SPBs
        spbs.loc[:, 'label'] = np.arange(spbs.shape[0])
        spbs.set_index('label', append=True, inplace=True)

        spbs.reset_index("main_label", inplace=True)

        solver = ByFrameSolver.for_brownian_motion(spbs, max_speed=v_max, coords=coords)
        spbs = solver.track(progress_bar=True)

        idx = pd.IndexSlice
        spbs.sortlevel(inplace=True)
        spbs.loc[idx[:, 0], 'side'] = 'A'
        spbs.loc[idx[:, 1], 'side'] = 'B'

        spbs.reset_index(level='label', inplace=True)

        # Do the merge
        spbs.set_index("main_label", append=True, inplace=True)
        peaks = pd.concat([kts, spbs]).sort_index()

        def label_kts(p):

            idx = pd.IndexSlice
            spb1 = p[p['side'] == 'A'].loc[idx[:, 'spb'], :]

            kt1 = p.loc[idx[:, 'kt'], :].iloc[0]
            kt2 = p.loc[idx[:, 'kt'], :].iloc[1]

            kt1_dist = np.linalg.norm(spb1[coords] - kt1[coords])
            kt2_dist = np.linalg.norm(spb1[coords] - kt2[coords])

            if kt1_dist < kt2_dist:
                p.loc[idx[:, 'kt'], 'side'] = np.array(['A', 'B'])
                p.loc[idx[:, 'kt'], 'label'] = np.array([2, 3])

            else:
                p.loc[idx[:, 'kt'], 'side'] = np.array(['B', 'A'])
                p.loc[idx[:, 'kt'], 'label'] = np.array([3, 2])

            return p

        peaks = progress_apply(peaks.groupby(level='t_stamp'), label_kts)
        peaks.set_index('side', append=True, inplace=True)

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
        peaks = Trajectories(self.peaks_real)

        self.switch_to_label(peaks)

        peaks = peaks.project([2, 3],
                              keep_first_time=False,
                              reference=None,
                              inplace=False,
                              progress=progress)

        self.unswitch_to_label(peaks)
        self.peaks_real = peaks

        log.info("*** End")

    def interpolate(self, sampling=1, s=0, k=1):
        """
        Interpolate columns. Then create new
        /peaks_real_interpolated with new times.
        """

        log.info("*** Running interpolate()")

        trajs = self.peaks_real.copy()
        self.switch_to_label(trajs)

        coords = list(trajs.columns)
        coords.remove('main_label')
        coords.remove('side')

        trajs = trajs.time_interpolate(sampling=sampling,
                                       s=s,
                                       k=k,
                                       coords=coords,
                                       keep_speed=False,
                                       keep_acceleration=False)

        self.unswitch_to_label(trajs)

        self.save(trajs, 'peaks_real_interpolated')

        log.info("*** End")

    def switch_to_label(self, peaks):
        """
        """

        if 'label' not in peaks.index.names:

            peaks['label'] = np.nan

            idx = pd.IndexSlice
            peaks.sortlevel(inplace=True)
            peaks.loc[idx[:, 'kt', 'A'], 'label'] = 0
            peaks.loc[idx[:, 'kt', 'B'], 'label'] = 1
            peaks.loc[idx[:, 'spb', 'A'], 'label'] = 2
            peaks.loc[idx[:, 'spb', 'B'], 'label'] = 3

            peaks.set_level_label(inplace=True)

    def unswitch_to_label(self, trajs):
        """
        """

        if 'label' in trajs.index.names:

            trajs['main_label'] = None
            trajs['side'] = None

            idx = pd.IndexSlice
            trajs.sortlevel(inplace=True)
            trajs.loc[idx[:, [0, 1]], 'main_label'] = 'kt'
            trajs.loc[idx[:, [2, 3]], 'main_label'] = 'spb'
            trajs.loc[idx[:, [0, 2]], 'side'] = 'A'
            trajs.loc[idx[:, [1, 3]], 'side'] = 'B'

            trajs.reset_index('label', inplace=True)
            trajs.set_index(['main_label', 'side'], append=True, inplace=True)
            trajs.sort_index(inplace=True)

    def get_marker(self, base_dir, suffix, d_th, dmax):
        """
        """
        #find trackmate XML file of the marker
        marker_xml_file = self.has_xml(self, suffix=suffix)
        marker_trackmate = trackmate_peak_import(os.path.join(self.base_dir, marker_xml_file))

        self.peaks_real[suffix + '_d'] = 0
        self.peaks_real[suffix + '_I'] = 0
        self.peaks_real[suffix + '_w'] = 0

        #fill peaks_real with marker dots properties when marker dots are colocalized with cen2
        for t_stamp, p in self.peaks_real.groupby(level=['t_stamp']):
            dA = dmax
            dB = dmax
            if t_stamp in marker_trackmate.index.get_level_values('t_stamp'):
                for i in range(len(marker_trackmate.loc[t_stamp]['x'])):
                    d1 = np.sqrt(((marker_trackmate.loc[t_stamp]['x'].values[i]- p.xs('A', level='side').xs('kt', level='main_label')['x'].loc[t_stamp]))**2 + ((marker_trackmate.loc[t_stamp]['y'].values[i]- p.xs('A', level='side').xs('kt', level='main_label')['y'].loc[t_stamp]))**2)
                    if ((d1 < d_th) & (d1 < dA)):
                        dA = d1
                        self.peaks_real.loc[(t_stamp, 'kt', 'A'), suffix +'_d'] = d1
                        self.peaks_real.loc[(t_stamp, 'kt', 'A'), suffix + '_I'] = marker_trackmate.loc[t_stamp]['I'].values[i]
                        self.peaks_real.loc[(t_stamp, 'kt', 'A'), suffix + '_w'] = marker_trackmate.loc[t_stamp]['w'].values[i]
                    d2 = np.sqrt(((marker_trackmate.loc[t_stamp]['x'].values[i]- p.xs('B', level='side').xs('kt', level='main_label')['x'].loc[t_stamp]))**2 + ((marker_trackmate.loc[t_stamp]['y'].values[i]- p.xs('B', level='side').xs('kt', level='main_label')['y'].loc[t_stamp]))**2)
                    if ((d2 < d_th) & (d2 < dB)):
                        self.peaks_real.loc[(t_stamp, 'kt', 'B'), suffix + '_d'] = d2
                        self.peaks_real.loc[(t_stamp, 'kt', 'B'), suffix + '_I'] = marker_trackmate.loc[t_stamp]['I'].values[i]
                        self.peaks_real.loc[(t_stamp, 'kt', 'B'), suffix + '_w'] = marker_trackmate.loc[t_stamp]['w'].values[i]

        self.save_oio()

        return self.peaks_real

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

        colors = list(Trajectories(peaks).get_colors().values())

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


    def kymo_coloc (self, suffix, use_interpolate=False, time_in_minutes=False):
        """
        """

        mpl_params_s = {'marker': 'o'}
        mpl_params = {'ls': '-'}
        minI = self.peaks_real[suffix + '_I'].loc[self.peaks_real[suffix +'_I'] != 0].min()

        if self.peaks_real.empty:
            log.error("peaks_real is empty")
            #return None

        if use_interpolate:
            peaks = self.peaks_real_interpolated
            times = self.times_interpolated
        else:
            peaks = self.peaks_real
            times = self.times


        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 7))
        ax = plt.subplot(111)
        drawer_s = ax.scatter
        drawer = ax.plot

        #peaks['is'+ suffix] = peaks[suffix + '_I']
        #peaks['is'+ suffix].loc[peaks['is'+ suffix]!=0] = 1

        gps = peaks.groupby(level=['main_label', 'side']).groups
        coord = 'x_proj'

        # Set axis limit
        ax.set_xlim(min(times), max(times))
        m = np.abs(peaks['x_proj'].max())
        ax.set_ylim(-m-0.2, m+0.2)


        # Draw SPBs and kts

        x = peaks.loc[gps[('spb', 'A')]][coord]
        drawer(times, x,  color='#8e8f99', **mpl_params)

        x = peaks.loc[gps[('spb', 'B')]][coord]
        drawer(times, x, label="SPBs", color='#8e8f99', **mpl_params)

        x = peaks.loc[gps[('kt', 'A')]][coord]
        drawer(times, x,  color='#a2a7ff', zorder=1, **mpl_params)

        x = peaks.loc[gps[('kt', 'B')]][coord]
        drawer(times, x, label="Kts", color='#a2a7ff', zorder=1 ,**mpl_params)

        #scatter SPB
        x = peaks.loc[gps[('spb', 'B')]][coord]
        drawer_s(times, x, color='#8e8f99', s = 40, **mpl_params_s)

        x = peaks.loc[gps[('spb', 'A')]][coord]
        drawer_s(times, x, color='#8e8f99', **mpl_params_s)


        #Get coloc indexs

        peaks['is'+ suffix] = peaks[suffix + '_I']
        peaks['is'+ suffix].loc[peaks['is'+ suffix]!=0] = 1

        peaks = peaks.set_index(['is'+ suffix], append = True, inplace = False)

        gps_kt = peaks.groupby(level=['main_label', 'side', 'is'+suffix]).groups

        #kt no coloc
        x = peaks.loc[gps_kt[('kt', 'A', 0)]][coord]
        times = peaks.loc[gps_kt[('kt', 'A', 0)]]['t']
        drawer_s(times, x, label="no " + suffix, color='#a2a7ff', **mpl_params_s)

        x = peaks.loc[gps_kt[('kt', 'B', 0)]][coord]
        times = peaks.loc[gps_kt[('kt', 'B', 0)]]['t']
        drawer_s(times, x, color='#a2a7ff', **mpl_params_s)


        # kt with coloc
        if len(peaks.xs('A', level='side').xs('kt', level='main_label').index.get_level_values('is'+suffix).unique()) > 1:
            x = peaks.loc[gps_kt[('kt', 'A', 1)]][coord]
            times = peaks.loc[gps_kt[('kt', 'A', 1)]]['t']
            I = peaks.loc[gps_kt[('kt', 'A', 1)]][suffix +'_I']
            marker_size = ((I/minI)**6)*30
            drawer_s(times, x, label="coloc " + suffix, color='#f00a0a', s= marker_size, zorder=2, **mpl_params_s)

        if  len(peaks.xs('B', level='side').xs('kt', level='main_label').index.get_level_values('is'+suffix).unique()) > 1:
            x = peaks.loc[gps_kt[('kt', 'B', 1)]][coord]
            times = peaks.loc[gps_kt[('kt', 'B', 1)]]['t']
            I = peaks.loc[gps_kt[('kt', 'B', 1)]][suffix + '_I']
            marker_size = ((I/minI)**6)*30
            drawer_s(times, x, color='#f00a0a',  s= marker_size, zorder=2, **mpl_params_s)


        fontsize = 22

        if time_in_minutes:
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

    def kymo_hfr(self, time_range=None, figsize=(12, 6), invert=False):
        """
        """
        import matplotlib.pyplot as plt
        import matplotlib

        if time_range:
            peaks = self.peaks_real_interpolated.loc[time_range[0]:time_range[1]]
            times = peaks['t'].unique()
        else:
            peaks = self.peaks_real_interpolated
            times = self.times_interpolated #/ 60

        times -= times[0]

        idx = pd.IndexSlice

        spbA = peaks.loc[idx[:, 'spb', 'A'], :]
        spbB = peaks.loc[idx[:, 'spb', 'B'], :]
        ktA = peaks.loc[idx[:, 'kt', 'A'], :]
        ktB = peaks.loc[idx[:, 'kt', 'B'], :]
        kts_traj = (ktA['x_proj'].values + ktB['x_proj'].values) / 2

        fig, ax = plt.subplots(figsize=figsize)

        kwargs = {'lw': 2}
        ax.plot(times, spbA['x_proj'], color='black', **kwargs)
        ax.plot(times, spbB['x_proj'], color='black', **kwargs)
        ax.plot(times, ktA['x_proj'], color='#ff2626', **kwargs)
        ax.plot(times, ktB['x_proj'], color='#ff2626', **kwargs)
        ax.plot(times, kts_traj, color='#00a0ff', **kwargs)

        ax.set_ylim(-4, 4)
        ax.set_yticks(np.arange(-4, 5, 2))

        ax.set_xlim(0, times[-1])
        ax.set_xticks(np.arange(0, times[-1], 50))

        nullform = matplotlib.ticker.FuncFormatter(lambda x, y: "")
        ax.xaxis.set_major_formatter(nullform)
        ax.yaxis.set_major_formatter(nullform)

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        for i in ax.spines.values():
            i.set_linewidth(4)
            i.set_color('black')

        ax.grid(b=True, which='major', color='#000000', linestyle='-', alpha=0.4, lw=2)
        ax.set_axisbelow(True)

        if invert:
            ax.invert_yaxis()

        return fig


    def kymo_directions(self, sister=True, base_score=0.15):
        """
        """

        times = self.times_interpolated_metaphase
        peaks = self.get_peaks_interpolated_metaphase()

        idx = pd.IndexSlice
        ktA = peaks.loc[idx[:, 'kt', 'A'], 'x_proj']
        ktB = peaks.loc[idx[:, 'kt', 'B'], 'x_proj']
        spbA = peaks.loc[idx[:, 'spb', 'A'], 'x_proj']
        spbB = peaks.loc[idx[:, 'spb', 'B'], 'x_proj']
        kts_traj = (ktA.values + ktB.values) / 2

        if sister:

            def color_run_traj(ax, times, traj, run_indexes, color, top=True):
                """
                """

                for t1, t2 in run_indexes:
                    x = times[t1:t2]
                    y1 = traj.values[t1:t2]
                    if top:
                        y2 = [10] * len(y1)
                    else:
                        y2 = [-10] * len(y1)
                    ax.fill_between(x, y1, y2, alpha=0.15, color=color)

            fig = self.kymo(mpl_params={'ls': '-', 'marker': ''})
            ax = fig.get_axes()[0]

            p, ap, _ = self.get_directions(ktA, window=10, base_score=base_score, side=-1,
                                           second=False, t0=0)
            color_run_traj(ax, times, ktA, p, 'g', top=True)
            color_run_traj(ax, times, ktA, ap, 'b', top=True)

            p, ap, _ = self.get_directions(ktB, window=10, base_score=base_score, side=1,
                                           second=False, t0=0)
            color_run_traj(ax, times, ktB, p, 'g', top=False)
            color_run_traj(ax, times, ktB, ap, 'b', top=False)

        else:

            p, ap, _ = self.get_directions(kts_traj, window=10, base_score=base_score, side=1,
                                           second=True, t0=times[0])

            fig = self.kymo(mpl_params={'ls': '-', 'marker': ''})
            ax = fig.get_axes()[0]

            ax.plot(times, kts_traj, color='black')

            for t1, t2 in p:
                ax.axvspan(t1, t2, facecolor='g', alpha=0.15)

            for t1, t2 in ap:
                ax.axvspan(t1, t2, facecolor='b', alpha=0.15)

        return fig

    def get_coherence(self, base_score=0.15):
        """
        N - N : 0
        N - P : 1
        N - AP : 2
        AP - P : 3
        P - P : 4
        AP - AP : 5
        """

        peaks = self.get_peaks_interpolated_metaphase()

        idx = pd.IndexSlice
        ktA = peaks.loc[idx[:, 'kt', 'A'], 'x_proj'].values
        ktB = peaks.loc[idx[:, 'kt', 'B'], 'x_proj'].values

        _, _, directionsA = self.get_directions(ktA, window=10, base_score=base_score, side=-1,
                                                second=False, t0=0)
        _, _, directionsB = self.get_directions(ktB, window=10, base_score=base_score, side=1,
                                                second=False, t0=0)

        m = np.char.add(directionsA, directionsB)

        # N - N
        m[(m == 'NNNN')] = 0

        # N - P
        m[(m == 'NNP') | (m == 'PNN')] = 1

        # N - AP
        m[(m == 'NNAP') | (m == 'APNN')] = 2

        # AP - P
        m[(m == 'APP') | (m == 'PAP')] = 3

        # P - P
        m[(m == 'PP')] = 4

        # AP - AP
        m[(m == 'APAP')] = 5

        coherence = m.astype('int')
        return coherence

    def link_runs(self, base_score=0.15):
        """
        """

        times = self.times_interpolated_metaphase
        peaks = self.get_peaks_interpolated_metaphase()

        idx = pd.IndexSlice
        ktA = peaks.loc[idx[:, 'kt', 'A'], 'x_proj'].values
        ktB = peaks.loc[idx[:, 'kt', 'B'], 'x_proj'].values

        p_A, ap_A, directions_A = self.get_directions(ktA, window=10, base_score=base_score,
                                                      side=-1, second=True, t0=times[0])
        p_B, ap_B, directions_B = self.get_directions(ktB, window=10, base_score=base_score,
                                                      side=1, second=True, t0=times[0])

        def find_nearest(array, value):
            idx = (np.abs(array - value)).argmin()
            return idx

        def link(run1, run2, start_time_offset=10, min_time=0.8):

            # # Check wether run1 or run2 contains too much interpolated timepoints
            # # above 50%
            # times = self.times_interpolated_metaphase
            # raw_times = self.times_metaphase

            # ## For run1
            # interp = times[find_nearest(times, run1[0]):find_nearest(times, run1[1])].shape[0]
            # no_interp = raw_times[find_nearest(raw_times, run1[0]):find_nearest(raw_times, run1[1])].shape[0]
            # interpolated_ratio = 1 - (no_interp / interp)

            # if interpolated_ratio > 0.6:
            #     return False

            # ## For run2
            # interp = times[find_nearest(times, run2[0]):find_nearest(times, run2[1])].shape[0]
            # no_interp = raw_times[find_nearest(raw_times, run2[0]):find_nearest(raw_times, run2[1])].shape[0]
            # interpolated_ratio = 1 - (no_interp / interp)

            # if interpolated_ratio > 0.6:
            #     return False

            # Check if they start at no more 'start_time_offset' interval
            if np.abs(run1[0] - run2[0]) > start_time_offset:
                return False

            # Check wether they overlap
            start = np.maximum(run1[0], run2[0])
            end = np.minimum(run1[1], run2[1])
            overlap_time = end - start

            if overlap_time < 0:
                return False

            # Check overlap time is more than 'min_time' of each run
            if (overlap_time / (run1[1] - run1[0])) < min_time:
                return False
            if (overlap_time / (run2[1] - run2[0])) < min_time:
                return False

            # Check wether run1 and run2 are longer than 10s
            if np.abs(run1[0] - run1[1]) < 10:
                return False
            if np.abs(run2[0] - run2[1]) < 10:
                return False

            return True

        def link_runs(indexes1, indexes2, start_time_offset=10, min_time=0.8):
            linked_run = []
            for run1, run2 in itertools.product(indexes1, indexes2):
                if link(run1, run2, start_time_offset=start_time_offset, min_time=min_time):
                    linked_run.append([run1.tolist(), run2.tolist()])

            return linked_run

        linked_runs = {}
        linked_runs['P-P'] = link_runs(p_A, p_B, start_time_offset=15, min_time=0.5)
        linked_runs['P-AP'] = link_runs(p_A, ap_B, start_time_offset=15, min_time=0.5)
        linked_runs['AP-AP'] = link_runs(ap_A, ap_B, start_time_offset=15, min_time=0.5)
        linked_runs['AP-P'] = link_runs(ap_A, p_B, start_time_offset=15, min_time=0.5)

        return linked_runs
