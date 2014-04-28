"""
Set of functions to easly select files recursively in a directory,
with matching patterns.
"""

import os
import re

__all__ = ['select_all', 'select_pattern', 'select_pattern_extract']


def select_all(main_path, discard=[],
               extensions=['.tif', '.tiff', '.TIF', '.TIFF'],
               no_match=None):
    """
    Select all fils matching with theses extensions in a given folder

    Parameters
    ----------
    main_path: string
        folder to look for files inside
    extensions: list of string
        Extensions to select

    """

    all_cropped = []

    for root, dirs, files in os.walk(main_path):
        parent_dir = os.path.basename(root)
        if parent_dir == 'cropped':
            cropped = os.listdir(root)
            for c in cropped:
                if os.path.splitext(c)[1] in extensions:
                    if no_match and no_match not in c:
                        all_cropped.append(os.path.join(root, c))

    all_cropped.sort()
    return all_cropped


def select_pattern(main_path, patterns, no_match=None):
    """
    Select file in folder which match with at least one of the pattern
    given in parameters
    """

    to_return = []

    patterns = [re.compile(p) for p in patterns]
    if no_match:
        d_no_match = lambda x: not (no_match in x)

    for root, dirs, files in os.walk(main_path):

        files = [os.path.join(root, f) for f in files]

        paths_match = []
        for p in patterns:
            paths_match += filter(p.match, files)

        to_return += paths_match

    if no_match:
        to_return = filter(d_no_match, to_return)

    to_return = list(set(to_return))
    to_return.sort()
    return to_return


def select_pattern_extract(main_path, patterns):
    """
    Select file in folder which match with at least one of the pattern
    given in parameters and extract variable into the given parameter
    """

    to_return = []

    patterns = [re.compile(p) for p in patterns]

    for root, dirs, files in os.walk(main_path):

        files = [os.path.join(root, f) for f in files]

        for f in files:
            for p in patterns:
                m = p.match(f)
                if m:
                    data = m.groupdict()
                    data['fname'] = f
                    to_return.append(data)

    return to_return
