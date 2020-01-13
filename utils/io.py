import os
from glob import glob as _glob


def glob(pathname, *, recursive=False, key=None, reverse=False):
    return sorted(_glob(pathname, recursive=recursive), key=key, reverse=reverse)


def prepare_path(*paths):
    path = os.path.join(*paths)
    dirpath = os.path.dirname(path)
    try:
        os.makedirs(dirpath)
    except:
        pass
    return path
