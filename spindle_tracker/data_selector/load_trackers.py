import os
import re
import logging

log = logging.getLogger(__name__)


from ..tracking import Cen2Tracker

from sktracker.utils.sort import natural_keys
from sktracker.utils import print_progress


def tracker_load(base_dir, movies_path, patterns, trackers=True):
    """
    """

    data_path = os.path.join(base_dir, movies_path)

    fnames = []

    # Select movies
    for root, dirs, files in os.walk(data_path):
        for f in files:
            fname = os.path.relpath(os.path.join(root, f), base_dir)
            if any([re.search(p, fname) for label, p in patterns]):
                fnames.append(fname)
    fnames = sorted(fnames, key=natural_keys)

    if not trackers:

        # Split by mutant groups
        gp = [(label, list(filter(lambda x: re.search(p, str(x)), fnames))) for label, p in patterns]
        gp = dict(gp)

        return fnames, gp

    log.info("Load {}".format(data_path))

    # Load trackers
    tracker_params = dict(base_dir=base_dir, verbose=False,
                          force_metadata=False, json_discovery=True)
    trackers = []
    for i, fname in enumerate(fnames):
        print_progress(i * 100 / len(fnames))
        tracker = Cen2Tracker(fname, **tracker_params)
        trackers.append(tracker)
    print_progress(-1)

    # Split by mutant groups
    trackers_gp = [(label, list(filter(lambda x: re.search(p, str(x)), trackers))) for label, p in patterns]
    trackers_gp = dict(trackers_gp)

    return fnames, trackers_gp
