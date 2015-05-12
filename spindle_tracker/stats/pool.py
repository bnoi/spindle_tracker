import itertools
import os

import numpy as np
from scipy import stats


def infos(labels, data, test_type="ks", save=None):
    """
    """

    max_label_size = max([len(label) for label in labels])

    s = ""

    s += "{:<{}} | {:<8} | {:<8} | {:<8} | {:<8}\n".format("label", max_label_size,
                                                           "n", "mean", "std", "sem")
    s += "\n"
    for label, d in zip(labels, data):
        m = "{:<{}} | {:<8} | {:<8.2e} | {:<8.2e} | {:<8.2e}\n"
        s += m.format(label, max_label_size, len(d), np.mean(d), np.std(d), stats.sem(d))

    s += "\n"

    if test_type == "ind":
        test_func = stats.ttest_ind
    elif test_type == "ks":
        test_func = stats.ks_2samp

    s += "{:<{}} | {:<8} | {:<8}\n".format("label", max_label_size * 2 + 3, "p-value", "test value")
    s += "\n"
    zipped = zip(list(itertools.combinations(labels, 2)), list(itertools.combinations(data, 2)))
    for (label1, label2), (data1, data2) in zipped:
        value, pvalue = test_func(data1, data2)
        m = "{:<{}} - {:<{}} | {:.2e} | {:.2e}\n"
        s += m.format(label1, max_label_size, label2, max_label_size, pvalue, value)

    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        with open(save, 'w') as f:
            f.write(s)

    return s
