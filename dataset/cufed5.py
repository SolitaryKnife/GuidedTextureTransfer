import torch
import torch.utils.data as data
import torchvision.transforms as T

from PIL import Image

import os.path as path

DATASET_SIZE = 126


class _IndexDataset(data.Dataset):
    def __init__(self, len_value, idx_transform):
        assert isinstance(len_value, int)
        assert len_value > 0
        assert callable(idx_transform)

        self.len_value = len_value
        self.idx_transform = idx_transform

    def __len__(self):
        return self.len_value

    def __getitem__(self, idx):
        assert isinstance(idx, int)
        if idx < 0 and idx >= self.len_value:
            raise IndexError(f"Index, {idx:03}, is out of range [0, {self.len_value})")

        return self.idx_transform(idx)


class Package(data.Dataset):

    def __init__(self, rootdir):
        assert isinstance(rootdir, str)

        self.rootdir = rootdir

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Current index ({idx}) exceeds length ({len(self)})")
        return tuple([path.join(self.rootdir, f"{idx:03}_{i}.png") for i in range(0, 6)])

    def __len__(self):
        return DATASET_SIZE

    def filepath_dataset(self, level=None, transform=None):
        if level is None:
            def load_filepath(idx):
                return self[idx]

        elif isinstance(level, int):
            assert level in range(0, 6)

            def load_filepath(idx):
                return self[idx][level]

        elif isinstance(level, (tuple, int)):
            assert all([(l in range(0, 6)) for l in level])

            def load_filepath(idx):
                return tuple([self[idx][l] for l in level])

        else:
            raise NotImplementedError()

        return _IndexDataset(DATASET_SIZE, T.Compose([
            load_filepath,
            transform or (lambda x:x)
        ]))

    def image_dataset(self, level=None, transform=None, image_loader=Image.open):
        assert transform is None or callable(transform)
        assert callable(image_loader)

        if level is None:
            def load_image(paths):
                return [image_loader(path) for path in paths]

        elif isinstance(level, int):
            assert level in range(0, 6)

            def load_image(path):
                return image_loader(path)

        elif isinstance(level, (tuple, int)):
            assert all([(l in range(0, 6)) for l in level])

            def load_image(paths):
                return [image_loader(path) for path in paths]

        else:
            raise NotImplementedError()

        return self.filepath_dataset(level=level, transform=T.Compose([
            load_image,
            transform or (lambda x:x)
        ]))
