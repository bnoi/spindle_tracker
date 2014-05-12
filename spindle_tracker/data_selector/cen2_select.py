import logging
import os
import datetime

import numpy as np

from sktracker.utils.progress import print_progress

from ..tracking import Cen2Tracker
from . import selector

log = logging.getLogger(__name__)

__all__ = ['cen2_select', 'get_last_data']


def cen2_select(data_path, base_dir, force_metadata=False):
    description = {'wt': {'patterns': ["^.*/.*tcx262/cropped/.*.tif$",
                                       "^.*/.*tcx263/cropped/.*.tif$",
                                       "^.*/.*tcx264/cropped/.*.tif$",
                                       "^.*/1237/cropped/.*.tif$"],
                          'metadata': {'name': 'wt',}
                          },
                  'klp5/6Δ': {'patterns': ["^.*/.*tcx265/cropped/.*.tif$",
                                          "^.*/1267/cropped/.*.tif$",
                                          "^.*/1266/cropped/.*.tif$"],
                             'metadata': {'name': 'klp5/6$Δ$',}
                             },
                  'dam1Δ': {'patterns': ["^.*/.*tcx266/cropped/.*.tif$",
                                        "^.*/.*tcx267/cropped/.*.tif$",
                                        "^.*/.*tcx268/cropped/.*.tif$",
                                        "^.*/1277/cropped/.*.tif$"],
                          'metadata': {'name': 'dam1$Δ$',}
                        }
                   }

    keys = ['wt', 'klp5/6Δ', 'dam1Δ']
    labels = []
    dataset = {}
    all_dataset = []

    def load_list_path(paths, base_dir):
        ret = []
        n = len(paths)
        for i, path in enumerate(paths):
            p = int(float(i + 1) / n * 100.)
            print_progress(p)
            relpath = os.path.relpath(path, base_dir)
            try:
                obj = Cen2Tracker(relpath, base_dir=base_dir, verbose=False,
                                  force_metadata=force_metadata)
                ret.append(obj)
            except:
                log.error("Error with {}".format(relpath))
        print_progress(-1)
        return ret

    for label in keys:
        values = description[label]

        dataset[label] = []
        log.info('Load dataset with label %s' % label)

        paths = selector.select_pattern(data_path, patterns=values['patterns'])
        dataset[label] = load_list_path(paths, base_dir)
        all_dataset.extend(dataset[label])
        labels.append(values['metadata']['name'])

    for label, values in dataset.items():
        nkymo = list(map(lambda x: x.annotations['kymo'], values))
        correct = list(filter(lambda x: x.annotations['kymo'] == 1 and x.annotations['anaphase'] != -1, values))

        nb_kymo = np.histogram(nkymo, bins=[0, 0.5, 1.5, 2.5])[0]

        spc = 15
        log.info("{0}: {1}".format("Label".rjust(spc), label))
        log.info('{0}: [ 0  1  2]'.format("Annotations ID".rjust(spc)))
        log.info("{0}: {1}".format("kymo".rjust(spc), nb_kymo))
        log.info("{0}: {1}".format("total".rjust(spc), len(values)))
        log.info("{0}: {1}".format("total correct".rjust(spc), len(correct)))
        log.info("")

    return dataset, all_dataset, keys, labels


def get_last_data(all_dataset, younger_than=2):

    def modification_date(filename):
        t = os.path.getmtime(filename)
        return datetime.datetime.fromtimestamp(t)

    def sort_by_date(tracker, younger_than):
        today = datetime.datetime.now()
        shift = datetime.timedelta(days=younger_than)
        younger_than = today - shift

        return modification_date(tracker.tif_path) > younger_than

    return list(filter(lambda x: sort_by_date(x, younger_than=younger_than), all_dataset))
