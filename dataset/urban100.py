import torch
import torch.utils.data as data

import torchvision
import torchvision.transforms as T

from glob import glob as _glob
from PIL import Image


def glob(pathname, *, recursive=False, key=None, reverse=False):
    return sorted(_glob(pathname, recursive=recursive), key=key, reverse=reverse)


class _ProxyDataset(data.Dataset):

    def __init__(self, proxy, transform=None):
        self.proxy = proxy
        self.transform = transform

    def __getitem__(self, idx):
        val = self.proxy[idx]
        if self.transform is not None:
            val = self.transform(val)
        return val

    def __len__(self):
        return len(self.proxy)


class Package(data.Dataset):

    def __init__(self, rootdir):
        assert isinstance(rootdir, str)
        self.lr = glob(f"{rootdir}/*_LR.png")
        self.hr = glob(f"{rootdir}/*_HR.png")

        assert len(self.lr) == len(self.hr)

    def __getitem__(self, idx):
        return self.lr[idx], self.hr[idx]

    def __len__(self):
        return len(self.hr)

    def filepath_dataset(self, transform=None):
        return _ProxyDataset(self, transform)

    def hr_filepath_dataset(self, transform=None):
        return _ProxyDataset(self, T.Compose([
            T.Lambda(lambda vals: vals[1]),
            transform or (lambda x:x)
        ]))

    def lr_filepath_dataset(self, transform=None):
        return _ProxyDataset(self, T.Compose([
            lambda vals: vals[0],
            transform or (lambda x:x)
        ]))

    def hr_image_dataset(self, transform=None, image_loader=Image.open):
        return self.hr_filepath_dataset(T.Compose([
            image_loader,
            transform or (lambda x:x)
        ]))

    def lr_image_dataset(self, transform=None, image_loader=Image.open):
        return self.lr_filepath_dataset(T.Compose([
            image_loader,
            transform or (lambda x:x)
        ]))

    def image_dataset(self, transform=None, image_loader=Image.open):
        return self.filepath_dataset(T.Compose([
            lambda vals: (image_loader(vals[0]), image_loader(vals[1])),
            transform or (lambda x:x)
        ]))
