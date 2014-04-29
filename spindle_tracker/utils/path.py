import os


def check_extension(file_path, extension, suffix=None, force=False):
    """
    Check if a given file (basename) as a same file with
    another check_extension. If force is True, new filename will be
    returned even if it does not exist.
    """

    basename = os.path.splitext(os.path.basename(file_path))[0]
    dirname = os.path.dirname(file_path)

    if suffix:
        new_name = '{}.{}.{}'.format(basename, suffix, extension)
    else:
        new_name = '{}.{}'.format(basename, extension)

    new_path = os.path.join(dirname, new_name)

    if os.path.isfile(new_path) or force:
        return new_path
    else:
        return None
