import numpy as np

def interp_nan(d):
    d = d.copy()
    ok = -np.isnan(d)
    xp = ok.ravel().nonzero()[0]
    fp = d[-np.isnan(d)]
    x = np.isnan(d).ravel().nonzero()[0]

    d[np.isnan(d)] = np.interp(x, xp, fp)
    return d
