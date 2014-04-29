import logging
import os

from dateutil import parser

from sktracker.io import TiffFile
from sktracker.io import StackIO
from sktracker.io import ObjectsIO
from sktracker.detection import peak_detector
from sktracker.io.trackmate import trackmate_peak_import

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

    def __init__(self,
                 sample_path,
                 base_dir,
                 verbose=True,
                 force_metadata=False,
                 json_discovery=True):
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

        # Init paths
        self.sample_path = sample_path
        self.base_dir = base_dir
        self.full_path = os.path.join(self.base_dir, self.sample_path)

        self.tif_path = self.has_tif()
        self.h5_path = self.has_h5()

        self.st = None
        self.oio = None

        if self.h5_path:
            self.oio = ObjectsIO.from_h5(self.h5_path,
                                         base_dir=self.base_dir,
                                         minimum_metadata_keys=self.__class__.MINIMUM_METADATA)
        elif self.tif_path:
            self.st = StackIO(image_path=self.tif_path,
                              base_dir=self.base_dir,
                              json_discovery=json_discovery)

            self.h5_path = self.has_h5(force=True)
            self.oio = ObjectsIO(self.st.metadata,
                                 store_path=self.h5_path,
                                 base_dir=self.base_dir,
                                 minimum_metadata_keys=self.__class__.MINIMUM_METADATA)
        else:
            raise IOError("h5 or/and tif file does not exist.")

        self.load_oio()

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
        if self.has_xml():
            return os.path.join(self.base_dir, self.has_xml())
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

    def has_xml(self, force=False):
        """
        """
        extension = self.__class__.XML_EXTENSION

        full_path = check_extension(self.full_path,
                                    extension=extension,
                                    force=force)

        if full_path:
            return os.path.relpath(full_path, self.base_dir)
        else:
            return None

    def load_oio(self):
        """
        """
        for key in self.oio.keys():
            key = key.replace('/', '')
            self.stored_data.append(key)
            setattr(self, key, self.oio[key])
            log.info("Correctly loaded '{}'".format(key))

    def save_oio(self):
        """
        """
        for key in self.stored_data:
            key = key.replace('/', '')
            self.oio[key] = getattr(self, key)
            log.info("Correctly saved '{}'".format(key))

        log.info("Data has been correctly saved to {}".format(self.h5_path))

    def get_tif(self):
        """
        """
        if self.full_tif_path:
            return TiffFile(self.full_tif_path)
        else:
            raise IOError("Tif path does not exist.")

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
            log.error("Can't parse or find acquisition date")
            return None

        return date_self < date_other

    def __gt__(self, other):
        return not self.__lt__(other)

    def detect_peaks(self,
                     detection_parameters,
                     channel=0,
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
                          json_discovery=False)

        data_iterator = self.st.image_iterator(channel_index=channel)

        peaks = peak_detector(data_iterator(),
                              self.metadata,
                              parallel=parallel,
                              show_progress=show_progress,
                              parameters=detection_parameters)

        self.stored_data.append('raw')
        self.raw = peaks

        self.save_oio()

    def get_peaks_from_trackmate(self, erase=False):
        """
        """

        if hasattr(self, 'raw') and not erase:
            log.info("Peaks already detected")
            return None

        xml_file = self.has_xml()
        if not xml_file:
            log.error("No XML file detected")

        self.raw = trackmate_peak_import(self.full_xml_path)
        self.stored_data.append('raw')
        self.save_oio()
