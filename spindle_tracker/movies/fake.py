import numpy as np
from scipy import signal

from tqdm import tqdm

from spindle_tracker.movies import psf
from spindle_tracker.io.tifffile import imsave


def fake_movie(peaks, save=None, noise_factor=5, border_factor=1.5):
    """
    """

    timepoints = peaks.index.get_level_values('t_stamp').unique()

    border_factor = 1.5
    noise_factor = 5

    w = np.round(np.abs(peaks['x'].min() - peaks['x'].max()), 0) * border_factor
    h = np.round(np.abs(peaks['y'].min() - peaks['y'].max()), 0) * border_factor

    z = np.round(np.abs(peaks['z'].min() - peaks['z'].max()), 0) * border_factor
    z = np.max([z, 5])

    t = np.round(timepoints[-1] - timepoints[0], 0) + 1

    # Make PFS
    args = dict(shape=(32, 32), dims=(4, 4), ex_wavelen=488, em_wavelen=520,
                num_aperture=1.2, refr_index=1.333,
                pinhole_radius=0.55, pinhole_shape='round')
    obsvol = psf.PSF(psf.GAUSSIAN | psf.CONFOCAL, **args)

    fakeim = np.zeros((t, z, w, h), dtype="float64")

    for t_stamp, spots in tqdm(peaks.groupby(level='t_stamp')):

        for idx, spot in spots.iterrows():

            x = spot['x'].round(0)
            y = spot['y'].round(0)
            z = spot['z'].round(0)
            w = spot['w'].round(0)

            if w % 2 == 0:
                border = w / 2 - 1
            else:
                border = w // 2

            fakeim[t_stamp, z, x - border:x + border, y - border:y + border] = 255

        fakeim[t_stamp] = signal.fftconvolve(fakeim[t_stamp], obsvol.volume(), mode="same")

    # Add some poisson noise
    noise = np.random.poisson(noise_factor * fakeim.max(), fakeim.shape)
    fakeim += noise

    fakeim = fakeim.astype("uint32")

    if save:
        imsave(save, fakeim)

    return fakeim
