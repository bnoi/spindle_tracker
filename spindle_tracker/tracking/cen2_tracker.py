from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import logging
import gc
import itertools
import dateutil

import pandas as pd
import numpy as np
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hier
from scipy.cluster import vq

from peak_detection import detect_peaks

from ..roi import read_roi_zip
from ..roi import read_roi
from ..roi.mask import get_mask
from ..tracking import Tracker
from ..io import TiffFile
from ..tracking.exception import *
from ..utils import id_generator
from ..utils.sci import interp_nan

log = logging.getLogger(__name__)

__all__ = ['Cen2Tracker', 'Cen2TrackerInitException']


class Cen2TrackerInitException(Exception):
    pass


class Cen2Tracker(Tracker):

    """
    Cen2Tracker is designed to track microscopic movies with cen2 and
    cdc11 labelled with GFP (cen2 refers to centromere which is close
    to Kinetochore, Kt, and cdc11 localized at the Spindle Pole Body,
    SPB).

    Notes
    -----
    Severals following methods should be moved to the parent class of
    Cen2Tracker, called Tracker, making it more generic.
    """

    MINIMUM_METADATA = ['x', 'y', 'z', 'c', 't',
                        'x-size', 'y-size', 'z-size',
                        'dt']

    def __init__(self, sample_path, verbose=True, force_metadata=False):
        """
        Parameters
        """

        self.analysis = {}

        Tracker.__init__(self, sample_path,
                         verbose=verbose,
                         force_metadata=force_metadata)

        # Create folder to store figure and data
        self.fig_path = os.path.join(self.sample_dir, "figures")

        self._enable_annotations()

        self._set_colors()

    def _set_colors(self):
        """
        Colors for plot
        """
        self.colors_cen2 = ["#A25540",  # SPB A
                            "#95BD55",  # SPB B
                            "#7F9C9A",  # Kt A
                            "#9E5EAB"]  # Kt B

        self.anaphase_col = "#A5507E"
        self.anaphase_before_col = "#9C603A"
        self.anaphase_after_col = "#8EBC78"
        self.kts_col = "#9BA24B"
        self.spbs_col = "#7D9DA5"

    def detect_peaks(self,
                     detection_params,
                     verbose=True,
                     show_progress=False,
                     parallel=True,
                     erase=False,
                     z_projection=False):
        """
        Wrapper to peak_detection. By default all channels, times and slices
        are given to peak_detection.
        Overwrite this function to change this.

        Parameters
        ----------
        detection_parameters : dict
            Parameters for peak_detection algorithm
        parallel: boolean, default False
            if True, runs detection in parallel
        erase: boolean, default False
            Erase .h5 and re-detect peaks if True
        """

        if isinstance(self.raw, pd.DataFrame) and not self.raw.empty and not erase:
            log.info("Peaks already detected.")
            return None

        if not self.tif_path or not os.path.isfile(self.tif_path):
            log.critical(
                "Can't detect peaks without Tiff file set in self.tif_path")
            return None

        # Scale w_s and peak_radius parameters
        detection_params = detection_params.copy()
        detection_params["w_s"] /= self.metadata['x-size']
        detection_params["w_s"] = int(np.round(detection_params["w_s"]))
        detection_params["peak_radius"] /= self.metadata['x-size']
        detection_params["peak_radius"] = np.round(detection_params["peak_radius"])

        sample = TiffFile(self.tif_path)

        curr_dir = os.path.dirname(__file__)
        fname = os.path.join(curr_dir, os.path.join(sample.fpath, sample.fname))

        log.info("Find peaks in %s" % fname)

        arr = sample.asarray()
        sample.close()

        # Try to get masked array from ROIs data
        mask = self.get_rois_mask(verbose=verbose)
        if isinstance(mask, np.ndarray):
            arr = np.ma.masked_array(arr, mask)

        # Remove single value dimensions
        single_idx = [i for i, s in enumerate(self.metadata['shape']) if s == 1]
        axes_names = [axe.lower() for i, axe in enumerate(self.metadata['axes']) if i not in single_idx]
        arr = np.squeeze(arr)

        # Detection only on GFP channel
        if 'c' in axes_names:
            channel_id = self.metadata['axes'].index('C')
            if self.metadata['shape'][channel_id] > 1:
                try:
                    gfp_id = self.metadata['channels'].index('GFP')
                except:
                    gfp_id = 0
                arr = np.take(arr, [gfp_id], axis=channel_id)
                arr = arr.squeeze(channel_id)
                axes_names.remove('c')
                arr = np.squeeze(arr)

        if z_projection and 'z' in axes_names:
            z_id = axes_names.index('z')
            arr = np.max(arr, axis=z_id)
            axes_names.remove('z')
            arr = np.squeeze(arr)

        peaks = detect_peaks(arr,
                             shape_label=axes_names,
                             verbose=verbose,
                             show_progress=show_progress,
                             parallel=parallel,
                             **detection_params)

        self.raw = peaks

        self.analysis['detected_peaks_mean'] = self.raw.groupby('t')['x'].count().mean()
        self.analysis['detected_peaks_std'] = self.raw.groupby('t')['x'].count().std()

        self.save_hdf5()

        del arr
        sample.close()
        del sample
        gc.collect()

    @property
    def times(self):
        return self.peaks_real.index.get_level_values('t').unique().astype('float')

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
                return int(np.argmin(np.abs(self.anaphase - peaks.index.get_level_values('t').values)) / 4)
            except ValueError:
                return None
        else:
            return None

    @property
    def index_anaphase_interpolated(self):
        if self.anaphase:
            peaks = self.peaks_real_interpolated
            return int(np.argmin(np.abs(self.anaphase - peaks.index.get_level_values('t').values)) / 4)
        else:
            return None

    def _enable_annotations(self):
        """
        """

        if not hasattr(self, 'annotations') or not isinstance(self.annotations, pd.Series):
            self.annotations = {'kymo': 0, 'anaphase': 0}
        elif hasattr(self, 'annotations') and isinstance(self.annotations, pd.Series):
            self.annotations = self.annotations.to_dict()
        else:
            log.error("Issue loading annotations")

        self.stored_data.append('annotations')

    def has_rois(self):
        """
        """
        rois_filename = os.path.splitext(self.tif_path)[0] + ".zip"
        roi_filename = os.path.splitext(self.tif_path)[0] + ".roi"

        rois = None
        if os.path.isfile(rois_filename):
            rois = read_roi_zip(rois_filename)
        elif os.path.isfile(roi_filename):
            rois = read_roi(roi_filename)

        return rois

    def get_rois_mask(self, verbose=False):
        """
        """

        rois = self.has_rois()

        if rois:
            log.info("ROI detected")
            log.info("Extracting mask")
            rois_mask = get_mask(self.metadata['shape'], rois, verbose=verbose)
            return rois_mask
        else:
            return None

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
                       min_intensities=0.2,
                       max_radius=0.3,
                       erase=False):
        """
        Parameter
        ---------

        min_intensities: float
            Between 0 - 1 (normalized intensities)
        max_radius: float
            Value in um
        """

        if hasattr(self, "peaks_z") and isinstance(self.peaks_z, pd.DataFrame) and not erase:
            return self.peaks_z

        peaks = self.peaks_z.copy()
        bads = []

        # min_intensity = peaks['I'].min()
        # range_intensity = peaks['I'].max() - peaks['I'].min()
        max_radius /= self.metadata['x-size']

        for t, pos in peaks.groupby('t'):
            # intensities = (pos['I'] - min_intensity) / range_intensity
            # bads.extend(pos[intensities < min_intensities].index.values)

            bads.extend(pos[pos['w'] > max_radius].index.values)

            if pos.shape[0] >= num_kept:
                bads.extend(pos.sort('I').iloc[:-num_kept].index.values)

        bads = list(set(bads))

        log.info('%i / %i peaks has been removed from peaks_z' %
                 (len(bads), len(peaks)))

        self.peaks_z = peaks.drop(bads)
        self.stored_data.append('peaks_z')

    def track(self, v_max, erase=False, round_time=False):
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

        if 'z' not in self.peaks_z.columns:
            self.peaks_z['z'] = 0

        self._set_real_coordinate()

        if round_time:
            self.peaks_real.reset_index(inplace=True)
            self.peaks_real['t'] = np.round(self.peaks_real['t']).astype('int')
            self.peaks_real.set_index(['t'], inplace=True)

        self._remove_uncomplete_timepoints()

        if not self.peaks_real.empty:
            self._label_peaks()
            self._label_peaks_side()
            if v_max:
                self._remove_outliers(v_max=v_max)
            self._project()
            self._interpolate()

            self.peaks_real = self.peaks_real.sort()
        else:
            log.error("peaks_real is empty")

    def _set_real_coordinate(self):
        """
        Try to find real x, y and z size and scale peaks_z to real coordinate.
        self.peaks_z is not modified. New DataFrame is created,
        self.peaks_real.
        Time_stamp is also modified to refer to real time found in TiffFile
        metadata.

        Returns
        -------
            self.peaks_real: DataFrame
        """

        log.info('Set real world coordinates and times')

        if 'z' not in self.peaks_z.columns:
            self.peaks_z['z'] = 0

        self.peaks_real = self.peaks_z.copy()
        self.peaks_real['x'] *= self.metadata['x-size']
        self.peaks_real['y'] *= self.metadata['y-size']

        if 'z-size' in self.metadata:
            self.peaks_real['z'] *= self.metadata['z-size']
        self.peaks_real['w'] *= self.metadata['x-size']

        if self.metadata['dt'] > 1:
            self.peaks_real['t'] = self.peaks_z['t'] * self.metadata['dt']
        else:
            log.warning("Timepoints not converted in seconds")

        # Remove old stacks-based index
        self.peaks_real.set_index(['t'], inplace=True)

        return self.peaks_real

    def _remove_uncomplete_timepoints(self, num_keep=4):
        """
        """

        peaks = self.peaks_real.copy()
        bads = []

        removed_t = 0
        for t, pos in peaks.groupby(level='t'):
            if len(pos) < num_keep:
                bads.extend(pos.index)
                removed_t += 1

        bads = list(set(bads))
        self.peaks_real = peaks.drop(bads)

        log.info('%i / %i uncomplete timepoints has been removed' %
                 (removed_t, len(np.unique(peaks.index.get_level_values('t').values))))

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

        for i, (t, peaks) in enumerate(self.peaks_real.groupby(level='t')):

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

        new_peaks_real.set_index('t', append=False, inplace=True)
        new_peaks_real.set_index('main_label', append=True, inplace=True)
        new_peaks_real = new_peaks_real.ix[:, ['x', 'y', 'z', 'w', 'I']]
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

        for t, peaks in peaks_real.groupby(level='t'):

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
                times = peaks.index.get_level_values('t').values

                for i in np.arange(start_t_idx, len(times)):
                    dt = times[i]

                    x = interp_nan(peaks.loc[:dt, 'x'])
                    y = interp_nan(peaks.loc[:dt, 'y'])
                    t = interp_nan(peaks.loc[:dt].index.get_level_values('t').values.astype('float'))

                    dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
                    t = np.diff(t)
                    v = dist / t

                    peaks.loc[int(times[1]):int(dt), 'v'] = v

                    nan_t = peaks.loc[:dt][peaks.loc[:dt]['v'] > v_max].index.get_level_values('t')

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
        for t, peaks in self.peaks_real.groupby(level='t'):
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

    def get_speed(self, use_interpolate=False):
        """
        """

        if use_interpolate and not hasattr(self, 'peaks_real_interpolated'):
            self._interpolate()

        if use_interpolate:
            peaks = self.peaks_real_interpolated
        else:
            peaks = self.peaks_real

        speeds = pd.DataFrame([])

        t = np.unique(peaks.index.get_level_values('t').values)
        dims = ['x', 'y', 'x_proj']

        for (label, side), p in peaks.groupby(level=['main_label', 'side']):
            new_pos = []

            new_pos.append(list(itertools.repeat(label, len(t))))
            new_pos.append(list(itertools.repeat(side, len(t))))
            new_pos.append(list(t))

            for d in dims:
                g = np.gradient(p[d].values, np.diff(t, n=0))
                new_pos.append(list(g))

            speeds = speeds.append(pd.DataFrame(new_pos).T)

        speeds.columns = ['main_label', 'side', 't'] + dims
        speeds = speeds.set_index(['t', 'main_label', 'side'])
        speeds = speeds.sort()

        speeds = speeds.astype('float')

        return speeds

    """
    Plot and visu methods
    """

    def interactive_view(self, image_indexes,
                         frame_indexes,
                         peaks_iterator,
                         scale_factor=1):
        """
        show_peaks_real = True
        if show_peaks_real:
            peaks_iterator = tracker.peaks_real.groupby(level='t')
            image_indexes = (tracker.times / tracker.metadata['dt']).astype('int')
            scale_factor = 1 / tracker.metadata['x-size']
            frame_indexes = tracker.times
        else:
            peaks_iterator = tracker.peaks_z.groupby('t')
            image_indexes = tracker.frames.astype('int')
            scale_factor = 1
            frame_indexes = tracker.frames

        tracker.interactive_view(image_indexes, frame_indexes, peaks_iterator, scale_factor)
        """

        tf = self.get_tiff()
        arr = tf.asarray()

        # Display GFP channel
        if 'channels' in self.metadata.keys():
            channel_id = self.metadata['axes'].index('C')
            gfp_id = self.metadata['channels'].index('GFP')
            arr = np.take(arr, [gfp_id], axis=channel_id)

        # Perform z projection
        if 'Z' in self.metadata['axes']:
            slice_id = self.metadata['axes'].index('Z')
            arr = arr.max(axis=slice_id)

        # Select timepoints
        frame_id = self.metadata['axes'].index('T')
        arr = np.take(arr, image_indexes, axis=frame_id)

        if 'channels' in self.metadata.keys():
            arr = arr.squeeze()

        image = arr

        super().interactive_view(frame_indexes,
                                 peaks_iterator,
                                 image,
                                 scale_factor)

    def kymo_path(self, scatter=False):
        """
        """
        ext = '_kymo.svg'
        if scatter:
            ext = '_kymo_scatter.svg'
        fpath = os.path.join(self.fig_path, self.sample_basename + ext)
        return fpath

    def kymo(self, use_interpolate=False, scatter=False,
             save=False, clear=True, erase=False, show=False,
             getaxes=False, time_in_minutes=True, scale_time=False):
        """
        Print a kymograph of DataFrame of peaks (self.peaks_real{'x_proj']})
        """

        if self.peaks_real.empty:
            log.error("peaks_real is empty")
            return None

        if use_interpolate:
            peaks = self.peaks_real_interpolated
        else:
            peaks = self.peaks_real

        if isinstance(save, str):
            fpath = save
        else:
            ext = '_kymo.svg'
            if scatter:
                ext = '_kymo_scatter.svg'
            fpath = os.path.join(self.fig_path, self.sample_basename + ext)
            if os.path.isfile(fpath) and not erase and not show:
                log.info('File already exists to %s' % fpath)
                return False

        if show:
            erase = False

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 7))
        ax = plt.subplot(111)

        drawer = ax.plot
        params = {}
        if scatter:
            drawer = ax.scatter
            params = {'s': 1}

        if scale_time:
            times = self.times * self.metadata['dt']
        else:
            times = self.times

        # Draw SPB
        x = peaks.xs('spb', level="main_label").xs(
            'A', level="side").ix[:, 'x_proj']
        drawer(times, x, label="SPB A",
               color=self.colors_cen2[0], **params)

        x = peaks.xs('spb', level="main_label").xs(
            'B', level="side").ix[:, 'x_proj']
        drawer(times, x, label="SPB B", color=self.colors_cen2[1], **params)

        # Draw Kt
        x = peaks.xs('kt', level="main_label").xs(
            'A', level="side").ix[:, 'x_proj']
        drawer(times, x, label="Kt A", color=self.colors_cen2[2], **params)

        x = peaks.xs('kt', level="main_label").xs(
            'B', level="side").ix[:, 'x_proj']
        drawer(times, x, label="Kt B", color=self.colors_cen2[3], **params)

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

        # if 'prometaphase_start' in self.metadata.keys() and self.metadata['prometaphase_start']:
        #     ax.axvline(x=self.metadata['prometaphase_start'],
        #                color='black',
        #                alpha=1,
        #                linestyle="-",
        #                label='Prometaphase start')

        if 'anaphase_start' in self.metadata.keys() and self.analysis['anaphase_start']:
            ax.axvline(x=self.analysis['anaphase_start'],
                       color='black',
                       alpha=1,
                       linestyle="--",
                       label='Anaphase start')

        # if 'anaphase_end' in self.metadata.keys() and self.metadata['anaphase_end']:
        #     ax.axvline(x=self.metadata['anaphase_end'],
        #                color='black',
        #                alpha=1,
        #                linestyle=":",
        #                label="Anaphase end")

        leg = ax.legend(loc='best', fancybox=True)
        leg.get_frame().set_alpha(0.5)

        plt.grid(True)
        plt.tight_layout()

        if getaxes:
            plt.close()
            return fig.get_axes()[0]
        elif save:
            plt.savefig(fpath)
            if clear:
                fig.clf()
                plt.close()
                import gc
                gc.collect()
        elif show:
            plt.show()
        else:
            # plt.close()
            pass
        return fig

    def get_unique_id(self):
        """
        """

        d = dateutil.parser.parse(self.metadata['acquisition_date'])
        date = d.strftime('%Y.%m.%d')
        bname = self.sample_basename
        uid = id_generator(8)
        return "%s_%s_%s" % (date, bname, uid)
