import itertools

import scipy as sp
import numpy as np


def print_stats(data,  abs=False):

    if abs:
        data = np.abs(data)

    groups = data.groupby(level=0)
    mean = groups.mean()
    std = groups.std()
    count = groups.count()
    sem = groups.aggregate(sp.stats.sem)

    print("Labels : {}".format(groups.mean().index.tolist()))
    print("N : {}".format(count.values.T[0]))
    print("Mean : {}".format(mean.values.T[0]))
    print("Std : {}".format(std.values.T[0]))
    print("SEM : {}".format(sem.values.T[0]))

    for label1, label2 in itertools.combinations(data.index.unique(), 2):
        print("Kolmogorov-Smirnov test : {} - {}".format(label1, label2))
        d1 = data.loc[label1].values.T[0]
        d2 = data.loc[label2].values.T[0]
        ks_value, p_value = sp.stats.ks_2samp(d1, d2)
        print("\tKS value = {}".format(ks_value))
        print("\tp-value = {:.2e}".format(p_value))
