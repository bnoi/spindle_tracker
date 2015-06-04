import itertools
import os

import numpy as np
from scipy import stats


def infos(labels, data, test_type="ks", save=None, do_resample_test=False, **kwargs):
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

    if not do_resample_test:

        s += "{:<{}} | {:<8} | {:<8}\n".format("label", max_label_size * 2 + 3, "p-value", "test value")
        s += "\n"
        zipped = zip(list(itertools.combinations(labels, 2)), list(itertools.combinations(data, 2)))
        for (label1, label2), (data1, data2) in zipped:
            value, pvalue = test_func(data1, data2)
            m = "{:<{}} - {:<{}} | {:.2e} | {:.2e}\n"
            s += m.format(label1, max_label_size, label2, max_label_size, pvalue, value)

    else:
        s += resample_test(labels, data, test_type=test_type, **kwargs)

    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        with open(save, 'w') as f:
            f.write(s)

    return s

def resample_test(labels, data, test_type="ks", pvalue_treshold=0.001, N=100, N_sample=100):
    """
    """
    max_label_size = max([len(label) for label in labels])

    if test_type == "ind":
        test_func = stats.ttest_ind
    elif test_type == "ks":
        test_func = stats.ks_2samp

    pvalues = {(label1, label2): [] for label1, label2 in itertools.combinations(labels, 2)}

    for i in range(N):
        sampled_data = []
        for dd in data:
            sampled_data.append(np.random.choice(dd, N_sample))

        zipped = zip(list(itertools.combinations(labels, 2)),
                     list(itertools.combinations(sampled_data, 2)))

        for (label1, label2), (data1, data2) in zipped:
            value, pvalue = stats.ttest_ind(data1, data2)
            pvalues[(label1, label2)].append(pvalue)

    pvalues = {k: np.array(v) for k, v in pvalues.items()}

    out = "Resample test for {} loop. Each loop randomly choose {} values in the dataset\n\n"
    out = out.format(N, N_sample)
    out += "{:<{}} | {:<8}\n".format("label", max_label_size * 2 + 3, "% below p-value of {}".format(pvalue_treshold))
    out += "\n"

    for (label1, label2), v in pvalues.items():
        above_treshold = int(np.sum(v <= pvalue_treshold) / N * 100)
        m = "{:<{}} - {:<{}} | {} %\n"
        out += m.format(label1, max_label_size, label2, max_label_size, above_treshold)

    return out
