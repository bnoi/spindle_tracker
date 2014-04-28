import logging
log = logging.getLogger(__name__)

import numpy as np
import pandas as pd

__all__ = ["get_fft", "get_fft_downsampled", "filter_signal"]


def get_fft(x, dt, hanning_window=False):
    n = len(x)

    if hanning_window:
        fft_output = np.fft.rfft(x * np.hamming(len(x)))
    else:
        fft_output = np.fft.rfft(x)

    rfreqs = np.fft.rfftfreq(n, d=dt)
    fft_mag = [np.sqrt(i.real ** 2 + i.imag ** 2) / n for i in fft_output]

    return np.array(fft_mag), np.array(rfreqs)


def get_fft_downsampled(x, dt, new_dt, verbose=False, hanning_window=False):
    """
    See http://www.dspguide.com/ch9/1.htm
    """

    n = x.size
    n_jump = new_dt / dt
    n_segments = n / n_jump

    if verbose:
        log.info("Numver of segments : {}".format(n_segments))
        log.info("Space between each element in one segment : {}".format(n_jump))
        log.info("New dt : {} s".format(new_dt))
        log.info("Max frequency : {} Hz".format(1 / new_dt))

    all_fft_mag = []
    all_rfreqs = []

    for i in np.arange(n_segments):
        new_x = x[i::n_jump]
        _fft_mag, _rfreqs = get_fft(new_x, new_dt, hanning_window=hanning_window)
        all_fft_mag.append(_fft_mag)
        all_rfreqs.append(_rfreqs)

    all_fft_mag = pd.DataFrame(all_fft_mag).values
    all_rfreqs = pd.DataFrame(all_rfreqs).values

    # Remove columns with Nan values
    all_fft_mag = np.ma.compress_rows(np.ma.fix_invalid(all_fft_mag.T)).T
    all_rfreqs = np.ma.compress_rows(np.ma.fix_invalid(all_rfreqs.T)).T

    # Get mean values along columns
    mean_fft_mag = all_fft_mag.mean(axis=0)
    mean_freqs = all_rfreqs.mean(axis=0)

    return mean_fft_mag, mean_freqs


def filter_signal(x, dt, filter_value, hanning_window=False):
    n = len(x)

    if hanning_window:
        fft_output = np.fft.rfft(x * np.hanning(x))
    else:
        fft_output = np.fft.rfft(x)

    rfreqs = np.fft.rfftfreq(n, d=dt)

    # Filtering
    fft_filtered = fft_output.copy()
    fft_filtered[(rfreqs > filter_value)] = 0

    # Inversed TF
    x_filtered = np.fft.irfft(fft_filtered)

    times = np.arange(0, n, dt)
    if x_filtered.size != times.size:
        times_filtered = times[:-1]
    else:
        times_filtered = times

    return x_filtered, times_filtered
