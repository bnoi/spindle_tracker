# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import logging
import json
import gc

import pandas as pd
import numpy as np
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hier
from scipy import interpolate

from sklearn.externals import joblib

from peak_detection import detect_peaks

from ..roi import read_roi_zip
from ..roi import read_roi
from ..roi.mask import get_mask
from ..tracking import Tracker
from ..tracking import check_data
from ..io import TiffFile
from ..tracking.exception import *

log = logging.getLogger(__name__)

__all__ = ['SPBTracker']






class SPBTracker(Tracker):


    MINIMUM_METADATA = ['x', 'y', 'z', 'c', 't',
                    'x-size', 'y-size',
                    'dt']

    HDF5_SUFFIX_FILE = "_detected_peaks"

    def __init__(self, sample_path, verbose=True, force_metadata=False):
        """
        Parameters
        """

        self.analysis = {}

        Tracker.__init__(self, sample_path,
                         verbose=verbose,
                         force_metadata=force_metadata)



