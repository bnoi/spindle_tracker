import itertools

import scipy as sp


def print_stats(data):

    groups = data.groupby(level=0)
    mean = groups.mean()
    std = groups.std()
    count = groups.count()
    sem = groups.aggregate(sp.stats.sem)

    print("Labels : {}".format(list(groups.groups.keys())))
    print("N : {}".format(count.values.T[0]))
    print("Mean : {}".format(mean.values.T[0]))
    print("Std : {}".format(std.values.T[0]))
    print("SEM : {}".format(sem.values.T[0]))

    for label1, label2 in itertools.combinations(data.index.unique(), 2):
        print("Kolmogorov-Smirnov test : {} - {}".format(label1, label2))
        ks_value, p_value = sp.stats.ks_2samp(data.loc[label1].values[0],
                                              data.loc[label2].values[0])
        print("\tKS value = {}".format(ks_value))
        print("\tp-value = {:.2e}".format(p_value))
