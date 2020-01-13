from . import data, io

from glob import glob as _glob


def glob(pathname, *, recursive=False, key=None, reverse=False):
    return sorted(_glob(pathname, recursive=recursive), key=key, reverse=reverse)


def cuda_all(*args):
    outputs = []
    for i, arg in enumerate(args):
        outputs.append(arg.cuda())
    return tuple(outputs)
