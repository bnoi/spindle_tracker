import logging
import os

from dateutil import parser

import pandas as pd

from sktracker.io import TiffFile
from sktracker.io import StackIO
from sktracker.io import ObjectsIO
from sktracker.detection import peak_detector
from sktracker.io.trackmate import trackmate_peak_import
from sktracker.trajectories import Trajectories
from sktracker.io import read_roi

from ..utils.short_id import id_generator
from ..utils.path import check_extension

log = logging.getLogger(__name__)


class Tracker():
    """
    Generic container for particle tracking
    """

    MINIMUM_METADATA = []
    HDF5_EXTENSION = "h5"
    TIF_EXTENSION = "tif"
    XML_EXTENSION = "xml"
    ANNOTATIONS = {}

    def __init__(self,
                 sample_path,
                 base_dir,
                 verbose=True,
                 force_metadata=False,
                 json_discovery=True,
                 clean_store=False):
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
        self._verbose = verbose

        # Init paths
        self.sample_path = sample_path
        self.base_dir = base_dir
        self.full_path = os.path.join(self.base_dir, self.sample_path)

        self.tif_path = self.has_tif()
        self.h5_path = self.has_h5()

        self.st = None
        self.oio = None

        if self.h5_path and not force_metadata:
            try:
                self.oio = ObjectsIO.from_h5(self.h5_path,
                                             base_dir=self.base_dir,
                                             minimum_metadata_keys=self.__class__.MINIMUM_METADATA,
                                             clean_store=clean_store)
            except ValueError as e:
                log.error(e)

        if self.tif_path and not self.oio:
            self.st = StackIO(image_path=self.tif_path,
                              base_dir=self.base_dir,
                              json_discovery=json_discovery)

            self.h5_path = self.has_h5(force=True)
            self.oio = ObjectsIO(self.st.metadata,
                                 store_path=self.h5_path,
                                 base_dir=self.base_dir,
                                 minimum_metadata_keys=self.__class__.MINIMUM_METADATA)

        if self.oio is None:
            raise IOError("Sample path should be a valid Tiff file or HDF5 file.")

        self.load_oio()
        self.setup_annotations()

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        log.disabled = not value
        self._verbose = value

    @property
    def full_tif_path(self):
        if self.tif_path:
            return os.path.join(self.base_dir, self.tif_path)
        else:
            return None

    @property
    def full_h5_path(self):
        if self.h5_path:
            return os.path.join(self.base_dir, self.h5_path)
        else:
            return None

    @property
    def full_xml_path(self):
        if self.xml_path:
            return os.path.join(self.base_dir, self.xml_path)
        else:
            return None

    @property
    def stored_data(self):
        if not hasattr(self, '_stored_data'):
            self._stored_data = []
        else:
            self._stored_data = list(set(self._stored_data))
        return self._stored_data

    def has_h5(self, force=False):
        """
        """
        extension = self.__class__.HDF5_EXTENSION

        full_path = check_extension(self.full_path,
                                    extension=extension,
                                    force=force)

        if full_path:
            return os.path.relpath(full_path, self.base_dir)
        else:
            return None

    def has_tif(self, force=False):
        """
        """
        extension = self.__class__.TIF_EXTENSION

        full_path = check_extension(self.full_path,
                                    extension=extension,
                                    force=force)

        if full_path:
            return os.path.relpath(full_path, self.base_dir)
        else:
            return None

    def has_xml(self, force=False, suffix=None):
        """
        """
        extension = self.__class__.XML_EXTENSION

        full_path = check_extension(self.full_path,
                                    extension=extension,
                                    suffix=suffix,
                                    force=force)

        if full_path:
            self.xml_path = os.path.relpath(full_path, self.base_dir)
            return self.xml_path
        else:
            return None

    def load_oio(self):
        """
        """
        for key, obj in self.oio.get_all_items():
            if isinstance(obj, pd.DataFrame):
                setattr(self, key, Trajectories(obj))
            else:
                setattr(self, key, obj)

            self.stored_data.append(key)
            log.info("Correctly loaded '{}'".format(key))

    def save_oio(self):
        """
        """
        for key in self.stored_data:
            self.oio[key] = getattr(self, key)

        log.info("Data has been correctly saved to {}".format(self.h5_path))

    def save(self, value, name=None):
        """Save an attribute to HDF5.
        """
        setattr(self, name, value)
        self.stored_data.append(name)
        self.save_oio()

    def get_tif(self, multifile=True):
        """
        """
        if self.full_tif_path:
            return TiffFile(self.full_tif_path, multifile=multifile)
        else:
            raise IOError("Tif path does not exist.")

    def __repr__(self):
        """
        """
        if self.has_tif():
            return self.has_tif()
        elif self.has_h5():
            return self.has_h5()
        else:
            return 'Error: No file found !'

    def __lt__(self, other):
        try:
            date_self = parser.parse(self.metadata['acquisition_date'])
            date_other = parser.parse(other.metadata['acquisition_date'])
        except:
            log.error("Can't parse or find acquisition date")
            return None

        return date_self < date_other

    def __gt__(self, other):
        return not self.__lt__(other)

    def detect_peaks(self,
                     detection_parameters,
                     channel=0,
                     z_projection=False,
                     show_progress=False,
                     parallel=True,
                     erase=False):
        """
        """

        if hasattr(self, 'raw') and not erase:
            log.info("Peaks already detected")
            return None

        if not self.full_tif_path or not os.path.isfile(self.full_tif_path):
            raise IOError("Tif path does not exist.")

        self.st = StackIO(image_path=self.tif_path,
                          base_dir=self.base_dir,
                          json_discovery=False,
                          metadata=self.metadata)

        data_iterator = self.st.image_iterator(channel_index=channel,
                                               z_projection=z_projection)

        if z_projection and 'Z' in self.metadata['DimensionOrder']:
            z_position = self.metadata['DimensionOrder'].index('Z')
            metadata = self.metadata.copy()
            metadata['Shape'] = list(metadata['Shape'])
            metadata['Shape'][z_position] = 1
            metadata['SizeZ'] = 1
        else:
            metadata = self.metadata

        peaks = peak_detector(data_iterator(),
                              metadata,
                              parallel=parallel,
                              show_progress=show_progress,
                              parameters=detection_parameters)

        self.stored_data.append('raw')
        self.raw = peaks

        self.save_oio()

    def get_peaks_from_trackmate(self, suffix=None):
        """
        """

        xml_file = self.has_xml(suffix=suffix)

        if not xml_file:
            log.warning("No Trackmate XML file detected.")
            return None

        self.raw_trackmate = trackmate_peak_import(self.full_xml_path)
        return self.raw_trackmate

    @property
    def unique_id(self):
        """
        """
        if 'unique_id' not in self.metadata.keys():
            self.metadata['unique_id'] = id_generator(6)
            self.save_oio()

        return self.metadata['unique_id']

    def setup_annotations(self):
        """
        """
        if 'annotations' not in self.stored_data:
            annotations = {}
            self.save(annotations, 'annotations')

        template = self.__class__.ANNOTATIONS
        save = False
        for k, (default, choices, value_type) in template.items():
            if k not in self.annotations.keys():
                self.annotations[k] = default
                save = True

        if save:
            self.save_oio()

    def open_roi(self, suffix='.ROI'):
        """Open ROI file (.zip and .roi)
        """

        roi_file_base = os.path.splitext(self.full_path)[0] + suffix
        if os.path.isfile(roi_file_base + '.zip'):
            return read_roi(roi_file_base + '.zip')
        elif os.path.isfile(roi_file_base + '.roi'):
            return read_roi(roi_file_base + '.roi')
        else:
            return None

    def show(self, var_name='trajs', marker='o', ls='-'):
        """
        """

        import matplotlib.pyplot as plt

        trajs = Trajectories(getattr(self, var_name))

        fig, (ax1, ax2) = plt.subplots(nrows=2)
        ax1 = trajs.show(xaxis='t', yaxis='x',
                         groupby_args={'level': "label"},
                         ax=ax1, ls=ls, marker=marker)

        ax2 = trajs.show(xaxis='t', yaxis='y',
                         groupby_args={'level': "label"},
                         ax=ax2, ls=ls, marker=marker)

        return fig
