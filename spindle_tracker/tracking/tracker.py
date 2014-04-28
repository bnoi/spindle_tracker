import logging

log = logging.getLogger(__name__)


class Tracker(object):
    """
    Generic container for particle tracking
    """

    MINIMUM_METADATA = []

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

        # # Check if Tiff file exist
        # self.tif_path = self.has_tif()
        # if not self.tif_path:
        #     log.warning('Tiff file does not exist.')

        # # Data to save in the h5 file
        # self.stored_data = ['metadata', 'raw']

        # self.h5_path = self.has_hdf5()
        # if not self.h5_path or not os.path.isfile(self.h5_path):
        #     log.error('HDF5 file is missing.')
        # if self.h5_path:
        #     self.load_hdf5()
        # if self.h5_path and ((isinstance(self.raw, pd.DataFrame) and self.raw.empty) or self.raw is None):
        #     log.warning('raw array is not present in HDF5 file.')
        #     log.warning('detect_peaks() should be called to create raw array.')

        # # Load metadata if not present in h5 file
        # if not isinstance(self.metadata, pd.Series) or force_metadata:
        #     self.metadata = self.load_metadata()
        #     self.check_metadata(self.metadata)
        #     self.save_hdf5()
        # else:
        #     # Convert Series to dict
        #     self.metadata = self.metadata.to_dict()
        #     self.check_metadata(self.metadata)

        # self.stored_data.append('analysis')
        # if isinstance(self.analysis, pd.Series):
        #     self.analysis = self.analysis.to_dict()
        # else:
        #     self.analysis = {}
