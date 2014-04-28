from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import logging
import json
import gc
from dateutil import parser

import pandas as pd
import numpy as np

from peak_detection import detect_peaks

from ..io import get_metadata_from_tiff
from ..io import TiffFile
from ..io import OMEModel
from .viewer import InteractiveView
from .exception import *
from ..segmentation import find_cell_boundary
from ..io import ij
from ..utils import id_generator

log = logging.getLogger(__name__)

__all__ = ['Tracker', 'check_data']


def check_data(*attr_names):
    """
    Usefull decorator to check attribute existence
    """

    def check_attr(obj, attr_name):
        if not hasattr(obj, attr_name) or \
           (hasattr(obj, attr_name) and not getattr(obj, attr_name)):
            raise Exception("Data missing: %s" % attr_name)

    def real_decorator(function):
        def wrapper(*args, **kwargs):
            obj = args[0]
            for attr_name in attr_names:
                check_attr(obj, attr_name)
            return function(*args, **kwargs)
        return wrapper
    return real_decorator


class Tracker(object):

    """
    Generic container for particle tracking
    """

    MINIMUM_METADATA = []

    HDF5_SUFFIX_FILE = "_detected_peaks"
    HDF5_EXTENSION = "h5"
    TIF_EXTENSION = "tif"

    def __init__(self, sample_path,
                 verbose=True,
                 force_metadata=False):
        """
        Parameters:
        -----------
        sample_path: string
            path to TIF file or to HDF5 file
        """

        if not verbose:
            log.disabled = True
        else:
            log.disabled = False

        # Init data
        self.metadata = None
        self.raw = None

        # Init paths
        sample_path = sample_path.replace(self.__class__.HDF5_SUFFIX_FILE, "")
        self.sample_dir = os.path.dirname(sample_path)
        self.sample_name = os.path.basename(sample_path)
        self.sample_basename = os.path.splitext(self.sample_name)[0]

        # Check if Tiff file exist
        self.tif_path = self.has_tif()
        if not self.tif_path:
            log.warning('Tiff file does not exist.')

        # Data to save in the h5 file
        self.stored_data = ['metadata', 'raw']

        self.h5_path = self.has_hdf5()
        if not self.h5_path or not os.path.isfile(self.h5_path):
            log.error('HDF5 file is missing.')
        if self.h5_path:
            self.load_hdf5()
        if self.h5_path and ((isinstance(self.raw, pd.DataFrame) and self.raw.empty) or self.raw is None):
            log.warning('raw array is not present in HDF5 file.')
            log.warning('detect_peaks() should be called to create raw array.')

        # Load metadata if not present in h5 file
        if not isinstance(self.metadata, pd.Series) or force_metadata:
            self.metadata = self.load_metadata()
            self.check_metadata(self.metadata)
            self.save_hdf5()
        else:
            # Convert Series to dict
            self.metadata = self.metadata.to_dict()
            self.check_metadata(self.metadata)

        self.stored_data.append('analysis')
        if isinstance(self.analysis, pd.Series):
            self.analysis = self.analysis.to_dict()
        else:
            self.analysis = {}

    def __repr__(self):
        """
        """
        if self.has_tif():
            return self.tif_path
        elif self.has_hdf5():
            return self.h5_path
        else:
            return 'Error: No file found !'

    def __lt__(self, other):
        try:
            date_self = parser.parse(self.metadata['acquisition_date'])
            date_other = parser.parse(other.metadata['acquisition_date'])
        except:
            print("Can't parse or find acquisition date")
            return None

        return date_self < date_other

    def __gt__(self, other):
        return not self.__lt__(other)

    def get_tiff(self):
        """
        """
        tf = TiffFile(self.tif_path)
        return tf

    def get_ome(self):
        """
        """
        tf = self.get_tiff()
        if tf.is_ome:
            p = tf.pages[0]
            xmlstr = p.tags['image_description'].value

            ome = OMEModel(xmlstr)
            return ome
        else:
            log.info('Tiff file does not contain OME metadata')
            return None

    def save_hdf5(self):
        """
        """

        self.stored_data = list(set(self.stored_data))

        self.h5_path = self.has_hdf5(force=True)

        store = pd.HDFStore(self.h5_path)
        for key in self.stored_data:
            attr = getattr(self, key)
            if isinstance(attr, dict):
                attr = pd.Series(list(attr.values()), index=attr.keys())
            if isinstance(attr, pd.DataFrame) or isinstance(attr, pd.Series):
                store[key] = attr
        store.close()

        log.info("Data has been correctly saved to %s " % self.h5_path)

    def load_hdf5(self):
        """
        Load data if not already loaded and present in h5 file.
        """

        self.stored_data = list(set(self.stored_data))

        store = pd.HDFStore(self.h5_path)
        for key in store.keys():
            try:
                attr_name = key.replace('/', '')
                setattr(self, attr_name, store[key])
                log.info("Correctly loaded '%s'" % attr_name)
                self.stored_data.append(attr_name)
            except:
                log.error("%s can't be loaded" % key)
                pass
        store.close()

    def check_extension(self, extension, suffix='', force=False):
        """
        Check if a given file (self.sample_basename) as a same file with
        another check_extension. If force is True, new filename will be
        returned even if it does not exist.
        """

        new_name = self.sample_basename + '%s.%s' % (suffix, extension)
        new_path = os.path.join(self.sample_dir, new_name)

        if os.path.isfile(new_path) or force:
            return new_path
        else:
            return None

    def has_hdf5(self, force=False):
        """
        """
        extension = self.__class__.HDF5_EXTENSION
        suffix = self.__class__.HDF5_SUFFIX_FILE

        return self.check_extension(extension=extension,
                                    suffix=suffix,
                                    force=force)

    def has_tif(self, force=False):
        """
        """
        extension = self.__class__.TIF_EXTENSION

        return self.check_extension(extension=extension,
                                    suffix="",
                                    force=force)

    def detect_peaks(self,
                     verbose=True,
                     show_progress=False,
                     detection_params=None,
                     parallel=True,
                     erase=False):
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

        if self.raw and not erase:
            log.info("Peaks already detected.")
        else:

            if not self.tif_path or not os.path.isfile(self.tif_path):
                log.critical(
                    "Can't detect peaks without Tiff file set in self.tif_path")
                return None

            if not detection_params:
                detection_params = {'w_s': 11,
                                    'peak_radius': 4.,
                                    'threshold': 40.,
                                    'max_peaks': 4
                                    }

            sample = TiffFile(self.tif_path)

            curr_dir = os.path.dirname(__file__)
            fname = os.path.join(
                curr_dir, os.path.join(sample.fpath, sample.fname))

            log.info("Find peaks in %s" % fname)

            arr = sample.asarray()
            shape = [x.lower() for x in list(sample.series[0]['axes'])]

            # TOFIX: peak_detection can only handle 4 dimensions for now
            if len(shape) == 5 and 'c' in shape:
                shape.remove('c')
                c_id = sample.series[0]['axes'].index('C')
                arr = arr.squeeze(c_id)

            sample.close()

            peaks = detect_peaks(arr,
                                 shape_label=shape,
                                 verbose=verbose,
                                 show_progress=show_progress,
                                 parallel=parallel,
                                 **detection_params)

            self.raw = peaks
            self.save_hdf5()

            del arr
            sample.close()
            del sample
            gc.collect()

        return self.raw

    def load_metadata(self):
        """
        Find metadata.json file in the same folder that tif or in the parent
        folder.
        """

        metadata = {}

        log.info('Load metadata from TiffFile')
        metadata_from_tiff = get_metadata_from_tiff(self.tif_path)
        metadata.update(metadata_from_tiff)

        candidats = [os.path.join(self.sample_dir, '..', 'metadata.json'),
                     os.path.join(self.sample_dir, 'metadata.json')]

        for metadata_path in candidats:
            if os.path.isfile(metadata_path):
                try:
                    log.info('Load metadata from %s' % metadata_path)
                    metadata.update(json.load(open(metadata_path)))
                except:
                    pass

        # Check if several channels are present
        # and raise an error if channel names are missing
        if 'c' in metadata.keys() and metadata['c'] > 1:
            if 'channels' not in metadata.keys():
                raise MetadataMissingException(
                    "Channel names metadata missing for %s" % self.tif_path)

        self.metadata = metadata
        return self.metadata

    def check_metadata(self, md):
        """
        Check metadata contains enough informations. If not raise an error.
        This function or self.__class__.MINIMUM_METADATA should be overriden by
        subclasses.
        """

        if not set(self.__class__.MINIMUM_METADATA).issubset(set(md.keys())):
            diff = set(self.__class__.MINIMUM_METADATA).difference(
                set(md.keys()))
            raise MetadataMissingException(
                'Some metadata missing: %s for %s' % (str(diff), self.tif_path))
        else:
            pass

    def interactive_view(self, frames_indexes, peaks_iterator, image, scale_factor=1):
        """
        """
        import matplotlib.pyplot as plt
        InteractiveView(
            self, frames_indexes, peaks_iterator, image, scale_factor)
        plt.show()

    def __del__(self):
        """
        """
        for d in self.stored_data:
            if hasattr(self, d):
                attr = getattr(self, d)
                del attr
        gc.collect()

    def get_boundaries(self,
                       object_height=3,
                       minimal_area=160,
                       channel_label="BF",
                       erase=False,
                       verbose=False):
        """
        Parameters
        ----------

        object_height : float
            Typical size of the object in um.
        minimal_area : float
            Typical area of the object in um^2.
        """

        if hasattr(self, "boundaries") and isinstance(self.boundaries, pd.DataFrame) and not erase:
            return self.boundaries

        tf = self.get_tiff()
        arr = tf.asarray()

        if not 'channels' in self.metadata.keys() or not 'axes' in self.metadata.keys():
            log.critical("Can't find 'channels' and/or 'axes' description in metadata")
            return None

        if 'C' not in self.metadata['axes']:
            log.critical("'C' dimension not found in 'axes' metadata")

        if channel_label not in self.metadata['channels']:
            log.critical("'%s' channel not found in 'channels' metadata" % channel_label)

        log.info("Get cell's boundaries with BF channel")

        channel_id = self.metadata['axes'].index('C')
        bf_id = self.metadata['channels'].index(channel_label)
        arr = np.take(arr, [bf_id], axis=channel_id)
        arr = arr.squeeze()

        sigma = object_height // self.metadata['z-size']
        minimal_area /= self.metadata['z-size']

        boundaries = find_cell_boundary(arr,
                                        sigma=sigma,
                                        minimal_area=minimal_area,
                                        verbose=verbose)

        if boundaries is None:
            log.error("No boundaries detected")
            return None

        boundaries['centroid_x'] *= self.metadata['x-size']
        boundaries['centroid_y'] *= self.metadata['x-size']
        boundaries['major_axis'] *= self.metadata['x-size']
        boundaries['minor_axis'] *= self.metadata['x-size']
        boundaries.index *= self.metadata['dt']

        self.boundaries = boundaries
        self.stored_data.append('boundaries')
        self.save_hdf5()

    def show_boundaries(self, channel_label="BF", z_projection=False, z=5):
        """
        """

        if channel_label not in self.metadata['channels']:
            log.critical("'%s' channel not found in 'channels' metadata" % channel_label)

        channel_id = self.metadata['axes'].index('C')
        bf_id = self.metadata['channels'].index(channel_label)

        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse

        b = self.boundaries.loc[0]
        x = b['centroid_x'] / self.metadata['x-size']
        y = b['centroid_y'] / self.metadata['x-size']
        major_axis = b['major_axis'] / self.metadata['x-size']
        minor_axis = b['minor_axis'] / self.metadata['x-size']
        orientation = b['orientation']
        orientation = np.rad2deg(orientation)

        arr = self.get_tiff().asarray()
        arr = np.take(arr, [bf_id], axis=channel_id).squeeze()

        if 'Z' in self.metadata['axes']:
            z_id = self.metadata['axes'].index('Z')
            if z_projection:
                arr = arr.max(axis=z_id - 1)
            else:
                arr = np.take(arr, [z], axis=z_id - 1).squeeze()
                print(arr.shape)

        a = arr[0]

        e = Ellipse(xy=(x, y), width=major_axis, height=minor_axis, angle=orientation)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_facecolor('red')
        e.set_alpha(0.5)

        ax.imshow(a, interpolation='none', cmap='gray')

        return fig

    def open_tif(self, ij_path):
        """
        """

        if self.has_tif():

            ij.set_ij_path(ij_path)
            p = ij.open_ome(self.tif_path)
            return p

        else:
            log.error("No Tiff file")
            return False

    def get_unique_id(self):
        """
        """

        bname = self.sample_basename
        uid = id_generator(8)
        return "%s_%s" % (bname, uid)
