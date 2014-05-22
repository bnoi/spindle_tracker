import re
import os
import logging

log = logging.getLogger(__name__)

from sktracker.utils.progress import print_progress

__all__ = ['tracker_select', 'rebuild_dict', 'merge_dict_of_list']


def tracker_select(paths, patterns, tracker_class, tracker_params, progress=False):
    """
    """
    # Select files
    combined_regex = "(" + ")|(".join(patterns) + ")"
    data_files = {}
    for label, path in paths.items():

        data_files[label] = []
        for root, dirs, files in os.walk(path):
            for f in files:
                f = os.path.join(root, f)
                if re.match(combined_regex, f):
                    data_files[label].append(f)

        log.info('{} files match with label {}'.format(len(data_files[label]), label))

    log.info("Total : {} files selected".format(sum(len(val) for val in data_files.values())))

    data_trackers = {}
    for label, path in data_files.items():

        data_trackers[label] = []
        n = len(path)

        log.info("Load {} tracker files for label {}".format(len(path), label))

        for i, f in enumerate(path):
            if progress:
                p = int(float(i + 1) / n * 100.)
                print_progress(p, message="{}/{}".format(i+1, n))

            tr = tracker_class(f, **tracker_params)
            data_trackers[label].append(tr)

        if progress:
            print_progress(-1)

    data_files_all = [item for sublist in data_files.values() for item in sublist]
    data_trackers_all = [item for sublist in data_trackers.values() for item in sublist]

    return data_trackers_all, data_trackers, data_files_all, data_files


def rebuild_dict(data_all, data):
    """
    """
    new_dict = {}
    for label, values in data.items():
        new_dict[label] = []
        for value in values:
            if value in data_all:
                new_dict[label].append(value)
        log.info("{}/{} trackers are selected".format(len(new_dict[label]), len(values)))

    return data_all, new_dict


def merge_dict_of_list(dol1, dol2):
    """
    """
    keys = set(dol1).union(dol2)
    no = []
    return dict((k, dol1.get(k, no) + dol2.get(k, no)) for k in keys)
