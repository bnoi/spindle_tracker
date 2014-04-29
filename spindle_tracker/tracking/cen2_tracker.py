import logging
import itertools
import os

import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hier
from scipy.cluster import vq

log = logging.getLogger(__name__)

from ..tracking import Tracker
from ..utils.sci import interp_nan


class Cen2Tracker(Tracker):

    MINIMUM_METADATA = ['SizeX', 'SizeY', 'SizeZ',
                        'PhysicalSizeX', 'PhysicalSizeY', 'PhysicalSizeZ',
                        'TimeIncrement']

    def __init__(self, *args, **kwargs):
        """
        """

        super().__init__(*args, **kwargs)
        self._enable_annotations()

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
            return self.analysis['anaphase_start']
        except:
            return None

    @property
    def index_anaphase(self):
        if self.anaphase:
            peaks = self.peaks_real
            try:
                return int(np.argmin(np.abs(self.anaphase - peaks['t'].values)) / 4)
            except ValueError:
                return None
        else:
            return None

    @property
    def index_anaphase_interpolated(self):
        if self.anaphase:
            peaks = self.peaks_real_interpolated
            return int(np.argmin(np.abs(self.anaphase - peaks['t'].values)) / 4)
        else:
            return None

    def _enable_annotations(self):
        """
        """

        if not hasattr(self, 'annotations') or not isinstance(self.annotations, pd.Series):
            self.annotations = {'kymo': 0}
        elif hasattr(self, 'annotations') and isinstance(self.annotations, pd.Series):
            self.annotations = self.annotations.to_dict()
        else:
            log.error("Issue loading annotations")

    """
    Tracking methods
    """

    def find_z_kmean(self, k=4, erase=False):
        """
        Find peaks with same x and y coordinate (with a cluster algorithm).
        Keep the peak with biggest intensity and add z coordinates
        according to his position in the z-stack.

        Parameters
        ----------
        k: int
            Number of cluster to detect.
        """
        if hasattr(self, "peaks_z") and isinstance(self.peaks_z, pd.DataFrame) and not erase:
            return self.peaks_z

        log.info("Find z relative coordinates with peaks"
                 " clustering max intensity selection.")

        peaks = self.raw.copy()
        bads = []

        for t, pos in peaks.groupby('t'):
            if pos.shape[0] > 1:
                centroids, _ = vq.kmeans(pos[['x', 'y']].values, k)
                clusters, _ = vq.vq(pos[['x', 'y']].values, centroids)
                pos['clusters'] = clusters
                for cluster, p in pos.groupby('clusters'):
                    bads.extend(p.sort(columns='I').iloc[:-1].index.values)

        self.peaks_z = peaks.drop(bads)
        self.stored_data.append('peaks_z')

    def find_z(self, treshold=5, erase=False):
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

        log.info("Find z relative coordinates with peaks"
                 " clustering max intensity selection.")

        peaks = self.raw.copy()
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

        self.peaks_z = peaks.drop(bads)
        self.peaks_z['clusters_count'] = clusters_count
        self.stored_data.append('peaks_z')

    def remove_weakers(self,
                       num_kept=4,
                       max_radius=0.3,
                       erase=False):
        """
        """

        if hasattr(self, "peaks_z") and isinstance(self.peaks_z, pd.DataFrame) and not erase:
            return self.peaks_z

        peaks = self.peaks_z.copy()
        bads = []

        for t, pos in peaks.groupby('t'):
            bads.extend(pos[pos['w'] > max_radius].index.values)

            if pos.shape[0] >= num_kept:
                bads.extend(pos.sort('I').iloc[:-num_kept].index.values)

        bads = list(set(bads))

        log.info('%i / %i peaks has been removed from peaks_z' %
                 (len(bads), len(peaks)))

        self.peaks_z = peaks.drop(bads)
        self.stored_data.append('peaks_z')

    def track(self, v_max, erase=False):
        """
        Set real coordinates. This process contains severals steps:
            1. Scale pixel coordinates to real world coordinates (µm)
            2. Remove timepoints with less than 4 peaks
            3. Label peaks (spb or Kt and side A or side B)
            4. Remove outliers peaks which exceed a given speed
            5. Project x, y coordinates to single axis (1,0)
               (x_proj in /peaks_real)
        """

        if hasattr(self, 'peaks_real') and isinstance(self.peaks_real, pd.DataFrame) and not erase:
            return self.peaks_real

        self.peaks_real = self.peaks_z.copy()
        self.stored_data.append('peaks_real')

        self._remove_uncomplete_timepoints()

        if not self.peaks_real.empty:
            self._label_peaks()
            self._label_peaks_side()
            # if v_max:
            #     self._remove_outliers(v_max=v_max)
            self._project()
            self._interpolate()

            self.peaks_real = self.peaks_real.sort()
        else:
            log.error("peaks_real is empty")

    def _remove_uncomplete_timepoints(self, num_keep=4):
        """
        """

        peaks = self.peaks_real.copy()
        bads = []

        removed_t = 0
        for t, pos in peaks.groupby('t'):
            if len(pos) < num_keep:
                bads.extend(pos.index)
                removed_t += 1

        bads = list(set(bads))
        self.peaks_real = peaks.drop(bads)

        log.info('%i / %i uncomplete timepoints has been removed' %
                 (removed_t, len(np.unique(peaks['t'].values))))

        return self.peaks_real

    def _label_peaks(self):
        """
        Labels SPB and Kt peaks

        Return
        ------
            self.peaks_real: DataFrame
        """

        log.info("Label SPB and Kt peaks")

        new_peaks_real = pd.DataFrame()
        # new_peaks_real['main_label'] = None

        for i, (t, peaks) in enumerate(self.peaks_real.groupby(level='t_stamp')):

            # Make a distance matrix and find the two peaks with
            # the biggest distance between them. This is our
            # SPB. We assume the two other peaks are Kt. Because z
            # coordinate is very inaccurate, we only use x and y
            # coordinate.
            peaks.reset_index(inplace=True)

            dist_mat = dist.squareform(dist.pdist(
                peaks.ix[:, ['x', 'y']]))

            spb_idx = list(np.unravel_index(dist_mat.argmax(),
                                            dist_mat.shape))
            spb_peaks = peaks.ix[spb_idx]
            spb_peaks['main_label'] = "spb"
            new_peaks_real = new_peaks_real.append(spb_peaks)

            kt_idx = set(range(len(peaks))) - set(spb_idx)
            kt_peaks = peaks.ix[list(kt_idx)]
            kt_peaks['main_label'] = "kt"
            new_peaks_real = new_peaks_real.append(kt_peaks)
            if len(kt_peaks) == 1:
                new_peaks_real = new_peaks_real.append(kt_peaks)

        new_peaks_real.set_index('t_stamp', append=False, inplace=True)
        new_peaks_real.set_index('main_label', append=True, inplace=True)
        new_peaks_real = new_peaks_real.ix[:, ['x', 'y', 'z', 'w', 'I', 't']]
        self.peaks_real = new_peaks_real

        self.stored_data.append('peaks_real')
        return self.peaks_real

    def _label_peaks_side(self,):
        """
        Label side for every couple peaks. Side is 'A' or 'B'.

        Return
        ------
            self.peaks_real: DataFrame
        """

        peaks_real = self.peaks_real
        log.info("Label peaks side")

        # Add id for each peaks
        # Note: rows id should be in default pandas DataFrame
        peaks_real['id'] = range(len(peaks_real))
        peaks_real.set_index('id', append=True, inplace=True)

        peaks_real['side'] = None

        for t, peaks in peaks_real.groupby(level='t_stamp'):

            spbs = peaks.xs('spb', level="main_label")
            kts = peaks.xs('kt', level="main_label")

            # Get center between the two SPB
            center = pd.concat([spbs[0:1], spbs[1:2]]).mean()

            # Setup vectors
            origin_vec = np.array([1, 0])
            x_shift = (spbs[0:1]['x'].values - spbs[1:2]['x'].values)[0]
            y_shift = (spbs[0:1]['y'].values - spbs[1:2]['y'].values)[0]

            if x_shift > y_shift:
                if spbs[0:1]['x'].values > spbs[1:2]['x'].values:
                    current_vec = (spbs[0:1]
                                   - center).ix[:, ('x', 'y')].values[0]
                else:
                    current_vec = (spbs[1:2]
                                   - center).ix[:, ('x', 'y')].values[0]
            else:
                if spbs[0:1]['y'].values > spbs[1:2]['y'].values:
                    current_vec = (spbs[0:1] -
                                   center).ix[:, ('x', 'y')].values[0]
                else:
                    current_vec = (spbs[1:2]
                                   - center).ix[:, ('x', 'y')].values[0]

            # Make coordinate with (,3) shape to allow dot product with
            # (3, 3) matrix (required for translation matrix)
            spbs_values = np.concatenate((spbs.ix[:, ('x', 'y')].values,
                                          np.array([[1, 1]]).T), axis=1)

            # Find the rotation angle
            cosa = np.dot(origin_vec,
                          current_vec) / (np.linalg.norm(origin_vec)
                                          * np.linalg.norm(current_vec))
            theta = np.arccos(cosa)
            # Build rotation matrix
            R = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]], dtype="float")

            # Build translation matrix
            T = np.array([[1, 0, -center['x']],
                          [0, 1, -center['y']],
                          [0, 0, 1]], dtype="float")

            # Make transformations from R and T in one
            A = np.dot(T.T, R)

            # Apply the transformation matrix
            spbs_new_values = np.dot(spbs_values, A)[:, 0]

            if spbs_new_values[0] < spbs_new_values[1]:
                spbA = spbs[0:1]
                spbB = spbs[1:2]
            else:
                spbA = spbs[1:2]
                spbB = spbs[0:1]

            spbA_id = spbA.index.get_level_values('id')[0]

            peaks_real.set_value((t, 'spb', spbA_id), 'side', 'A')
            spbB_id = spbB.index.get_level_values('id')[0]
            peaks_real.set_value((t, 'spb', spbB_id), 'side', 'B')

            # Compute the distance between spb A and the two Kt
            # Kt which is the closest to spbA is labeled to KtA
            # and the other to spbB

            # We assume there is two Kt

            dist1 = np.linalg.norm(kts[0:1].ix[:, ('x', 'y')].values
                                   - spbA.ix[:, ('x', 'y')].values)
            dist2 = np.linalg.norm(kts[1:2].ix[:, ('x', 'y')].values
                                   - spbA.ix[:, ('x', 'y')].values)

            if dist1 < dist2:
                ktA_id = kts[0:1].index.get_level_values('id')[0]
                ktB_id = kts[1:2].index.get_level_values('id')[0]
            else:
                ktA_id = kts[1:2].index.get_level_values('id')[0]
                ktB_id = kts[0:1].index.get_level_values('id')[0]

            peaks_real.set_value((t, 'kt', ktA_id), 'side', 'A')
            peaks_real.set_value((t, 'kt', ktB_id), 'side', 'B')

        peaks_real.reset_index(level='id', inplace=True)
        peaks_real.set_index('side', append=True, inplace=True)

        self.peaks_real = peaks_real
        return self.peaks_real

    def _remove_outliers(self, v_max):
        """
        Remove outliers peaks which overtake a speed given as a parameter.
        self.peaks_real is overriden

        Parameters
        ----------
        v_max: float
            maximum speed for one peak (in µm/s)
        """

        log.info("Remove outliers from peaks_real")

        new_peaks = pd.DataFrame()

        for (main_label, side), peaks in self.peaks_real.groupby(level=('main_label', 'side')):
            out = False
            start_t_idx = 2

            while not out:

                peaks['v'] = np.nan
                times = peaks['t'].values

                for i in np.arange(start_t_idx, len(times)):
                    dt = times[i]
                    print(peaks)
                    x = interp_nan(peaks.loc[:dt, 'x'])
                    y = interp_nan(peaks.loc[:dt, 'y'])
                    t = interp_nan(peaks.loc[:dt, 't'])

                    dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
                    t = np.diff(t)
                    v = dist / t

                    peaks.loc[int(times[1]):int(dt), 'v'] = v

                    nan_t = peaks.loc[:dt][peaks.loc[:dt]['v'] > v_max]['t']

                    # If outliers detected
                    if nan_t.any():
                        start_t_idx = i + 1
                        for t in nan_t:
                            peaks.loc[t] = np.nan
                        break

                if dt == times[-1]:
                    out = True

            peaks = peaks.dropna()  # or do interpolation
            # peaks['x'] = interp_nan(peaks['x'])
            # peaks['y'] = interp_nan(peaks['y'])

            new_peaks = new_peaks.append(peaks.reset_index())

        new_peaks.set_index(['t', 'main_label', 'side'], inplace=True)
        new_peaks.sort(inplace=True)

        self.peaks_real = new_peaks
        self._remove_uncomplete_timepoints()

        return self.peaks_real

    def _project(self, erase=False):
        """
        Project peaks from 2D to 1D taking SPB - SPB for the main axis.

        Returns
        -------
            self.peaks_real: DataFrame
        """

        log.info("Project peaks from 2D to 1D defined by SPB - SPB axis.")
        self.peaks_real['x_proj'] = None
        for t, peaks in self.peaks_real.groupby(level='t_stamp'):
            spbs = peaks.loc[t].loc['spb'].loc[:, ('x', 'y')]
            kts = peaks.loc[t].loc['kt'].loc[:, ('x', 'y')]

            spbA = spbs.xs('A')
            spbB = spbs.xs('B')

            # Get center between the two SPB
            center = pd.concat([spbA, spbB], axis=1).T.mean()
            # Setup vectors
            origin_vec = np.array([1, 0])
            current_vec = (spbB - center)
            # Make coordinate with (,3) shape to allow dot product with
            # (3, 3) matrix (required for translation matrix)
            spbs_values = np.concatenate((spbs.values,
                                          np.array([[1, 1]]).T), axis=1)
            kts_values = np.concatenate((kts.values,
                                         np.array([[1, 1]]).T), axis=1)
            # Find the rotation angle
            cosa = np.dot(origin_vec, current_vec) / (
                np.linalg.norm(origin_vec) * np.linalg.norm(current_vec))
            theta = np.arccos(cosa)

            # Build rotation matrix
            R = np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]], dtype="float")

            # Build translation matrix
            T = np.array([[1, 0, -center['x']],
                          [0, 1, -center['y']],
                          [0, 0, 1]], dtype="float")

            # Make transformations from R and T in one
            A = np.dot(T.T, R)

            # Apply the transformation matrix
            spbs_new_values = np.dot(spbs_values, A)[:, :-1]
            kts_new_values = np.dot(kts_values, A)[:, :-1]

            # Project Kt peaks on SPB - SPB right
            # For H, projection of A on right with v vector
            # Projection equation is: v * (dot(A, v) / norm(v) ** 2)
            # kts_proj_values = origin_vec * np.array([(
            #     np.dot(kts_new_values, origin_vec)
            #     / np.linalg.norm(origin_vec) ** 2)]).T

            self.peaks_real.set_value((t, 'spb', spbs.index[0]),
                                      'x_proj', spbs_new_values[:, 0][0])
            self.peaks_real.set_value((t, 'spb', spbs.index[1]),
                                      'x_proj', spbs_new_values[:, 0][1])

            self.peaks_real.set_value((t, 'kt', kts.index[0]),
                                      'x_proj', kts_new_values[:, 0][0])
            self.peaks_real.set_value((t, 'kt', kts.index[1]),
                                      'x_proj', kts_new_values[:, 0][1])

        self.peaks_real['x_proj'] = self.peaks_real['x_proj'].astype('float')
        return self.peaks_real

    def _interpolate(self, dt=1):
        """
        Interpolate data for x, y, z and x_proj. Then create new
        /peaks_real_interpolated with new times.
        """

        log.info('Interpolate peaks coordinates')

        peaks = self.peaks_real

        peaks = self.peaks_real
        peaks_interp = pd.DataFrame([])

        t = self.times
        new_t = np.arange(0, t.max(), dt)
        dims = ['x', 'y', 'x_proj']

        for (label, side), p in peaks.groupby(level=['main_label', 'side']):
            new_pos = []

            new_pos.append(list(itertools.repeat(label, len(new_t))))
            new_pos.append(list(itertools.repeat(side, len(new_t))))
            new_pos.append(list(new_t))

            for d in dims:
                new_pos.append(list(np.interp(new_t, t, p[d])))

            peaks_interp = peaks_interp.append(pd.DataFrame(new_pos).T)

        peaks_interp.columns = ['main_label', 'side', 't'] + dims
        peaks_interp = peaks_interp.set_index(['t', 'main_label', 'side'])
        peaks_interp = peaks_interp.sort()
        peaks_interp = peaks_interp.astype('float')

        self.stored_data.append('peaks_real_interpolated')
        self.peaks_real_interpolated = peaks_interp
        return self.peaks_real_interpolated

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

        # Draw SPB
        x = peaks.loc[gps[('spb', 'A')]]['x_proj']
        drawer(times, x, label="SPB A", color=colors[0], **mpl_params)

        x = peaks.loc[gps[('spb', 'B')]]['x_proj']
        drawer(times, x, label="SPB B", color=colors[1], **mpl_params)

        # Draw Kt
        x = peaks.loc[gps[('kt', 'A')]]['x_proj']
        drawer(times, x, label="Kt A", color=colors[2], **mpl_params)

        x = peaks.loc[gps[('kt', 'B')]]['x_proj']
        drawer(times, x, label="Kt B", color=colors[3], **mpl_params)

        # Set axis limit
        ax.set_xlim(min(times), max(times))
        ax.set_ylim(-6, 6)

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
