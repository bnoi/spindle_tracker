import argparse
import os
import logging
import tempfile
import shutil
import subprocess
import logging

import numpy as np

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from ..io import TiffFile
from ..utils.progress import print_progress

log = logging.getLogger(__name__)


def resize_same_aspect(img, size):
    """
    """
    wpercent = (size[0] / img.size[0])
    hsize = int((img.size[1] * wpercent))
    img = img.resize((size[0], hsize), Image.ANTIALIAS)
    return img


def draw_text(img, text, color="#ffffff", font_path="arial.ttf"):
    """
    """
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, int(0.1 * img.size[1]))

    text_w, text_h = draw.textsize(text, font)
    text_x = int(img.size[0] - text_w - img.size[0] * 0.1)
    text_y = int(img.size[1] - text_h - img.size[1] * 0.1)

    draw.text((text_x, text_y), text, font=font, fill=color)
    return img


def get_time(i, time_per_frame, max_digits, minute=True):
    """
    """
    t = i * time_per_frame
    if minute:
        t = int(t / 60)
        suffix = "min"
    else:
        suffix = "s"

    t = np.round(t, 1)

    ret = "%s%s" % (str(t).rjust(max_digits), suffix)
    return ret


def divisible_by_2(arr):
    """
    """
    arr = np.atleast_3d(arr)

    # Dimensions must be divisible by 2 (libx264)
    arr = arr.swapaxes(2,0).swapaxes(2,1)
    new_arr = []
    for a in arr:
        if (a.shape[0] % 2) != 0:
            a = np.insert(a, 0, np.zeros(a.shape[1]), axis=0)
        if (a.shape[1] % 2) != 0:
            a = np.insert(a, 0, np.zeros(a.shape[0]), axis=1)
        new_arr.append(a)

    new_arr = np.array(new_arr)
    new_arr = np.squeeze(new_arr.T)
    new_arr = np.asarray(new_arr)

    return new_arr


def create(input, output, fps=25, spf=10,
           resize=None, annotate=None, codec='mjpeg',
           gif=False, rgb=False, z_index=None,
           channel_order=None):
    """
    """

    font_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "arial.ttf")

    log.info("Load tiff file")
    t = TiffFile(input)
    arr = t.asarray()
    arr = np.squeeze(arr)

    log.info("Convert tiff to 8bit")
    arr = np.round(255.0 * (arr - arr.min()) / (arr.max() - arr.min() - 1.0)).astype(np.uint8)

    if rgb:
        if len(arr.shape) == 5:
            if z_index:
                arr = arr.max(axis=z_index)
            else:
                log.critical("You need to provide a z_index.")
        if len(arr.shape) != 4:
            log.critical("To use RGB Tiff, array needs to have 4 dimensions.")
    else:
        if len(arr.shape) == 4:
            log.info("Perform maximum projection on second axis")
            arr = arr.max(axis=1)
        if len(arr.shape) != 3:
            raise Exception("Tif array shape is not valid. It should be 3 or 4 dimensions.")

    i_max = arr.shape[0]
    frames_max_digits = int(np.log10(i_max)) + 1
    time_max_digits_second = int(np.log10(i_max * spf)) + 1
    time_max_digits_minute = int(np.log10(i_max * spf / 60)) + 1

    frames_dir = tempfile.mkdtemp()
    frames_pattern = os.path.join(frames_dir, "frame_") + "%%0%id.png" % frames_max_digits
    log.info("Generate {} image frames from Tiff in {}".format(arr.shape[0], frames_dir))

    if arr.shape[1] == 2:
        arr = np.insert(arr, 0, np.zeros(arr.shape[-2:]), axis=1)

    if channel_order:
        arr[:, [0, 1, 2]] = arr[:, channel_order]

    n = len(arr)
    for i, a in enumerate(arr):
        print_progress(i*100/n)

        if rgb:
            im = Image.merge('RGB', (Image.fromarray(a[0], 'L'),
                                     Image.fromarray(a[1], 'L'),
                                     Image.fromarray(a[2], 'L')))
        else:
            im = Image.fromarray(a, 'L')

        if resize:
            im = resize_same_aspect(im, (resize, resize))

        if annotate:
            if annotate == 's':
                text = get_time(i, spf, time_max_digits_second, minute=False)
            else:
                text = get_time(i, spf, time_max_digits_minute, minute=True)
            draw_text(im, text, color="#ffffff", font_path=font_path)

        frame_name = "frame_%s.png" % str(i).zfill(frames_max_digits)
        frame_path = os.path.join(frames_dir, frame_name)

        a = divisible_by_2(np.array(im))

        if a.ndim == 3:
            im = Image.merge('RGB', (Image.fromarray(a.T[0], 'L'),
                                     Image.fromarray(a.T[1], 'L'),
                                     Image.fromarray(a.T[2], 'L')))
        else:
            im = Image.fromarray(a.T, 'L')

        im.save(frame_path)
        del im

    print_progress(-1)

    if gif:
        log.info("Generate GIF file with 'convert'")
        cmd = "convert -delay {} -loop 0 {}/*.png {}".format(fps, frames_dir, output)
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        log.info("Generate movie with 'avconv'")
        cmd = "avconv -y -r {} -f image2 -i {} -vcodec {} {}".format(fps, frames_pattern, codec, output)
        p = subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        log.info("Reencode at 30 fps")
        tmp_output = os.path.join(tempfile.mkdtemp(), os.path.basename(output))
        cmd = "avconv -i {} -r 30 -codec:a copy {}".format(output, tmp_output)
        p = subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        shutil.copy(tmp_output, output)
        shutil.rmtree(os.path.dirname(tmp_output))

    log.info("Clean image frames in {}".format(frames_dir))
    shutil.rmtree(frames_dir)

    log.info('Movie generated to {}'.format(output))
