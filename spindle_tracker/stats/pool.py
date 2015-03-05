import itertools

import numpy as np
from scipy import stats


def infos(labels, data, test_type="ks"):
    """
    """

    s = ""

    s += "{:<16} | {:<8} | {:<8} | {:<8} | {:<8}\n".format("label", "n", "mean", "std", "sem")
    s += "\n"
    for label, d in zip(labels, data):
        m = "{:<16} | {:<8} | {:<8.2e} | {:<8.2e} | {:<8.2e}\n"
        s += m.format(label, len(d), np.mean(d), np.std(d), stats.sem(d))

    s += "\n"

    if test_type == "ind":
        test_func = stats.ttest_ind
    elif test_type == "ks":
        test_func = stats.ks_2samp

    s += "{:<30} | {:<8} | {:<8}\n".format("label", "p-value", "test value")
    s += "\n"
    zipped = zip(list(itertools.combinations(labels, 2)), list(itertools.combinations(data, 2)))
    for (label1, label2), (data1, data2) in zipped:
        value, pvalue = test_func(data1, data2)
        m = "{:<12} - {:<15} | {:.2e} | {:.2e}\n"
        s += m.format(label1, label2, pvalue, value)

    return s
