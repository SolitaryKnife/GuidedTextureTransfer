import torch
import torchvision
import torchvision.transforms.functional as TF

import numpy as np

from PIL import Image

RESIZE_MODES = {
    None: 0,
    "NEAREST": 0,
    "NONE": 0,
    "BOX": 4,
    "BILINEAR": 2,
    "LINEAR": 2,
    "HAMMING": 5,
    "BICUBIC": 3,
    "CUBIC": 3,
    "LANCZOS": 1,
    "ANTIALIAS": 1,
}


def load_image(path, *, resize_scale=None, resize_size=None, resize_mode="bicubic"):
    assert isinstance(path, str)

    if resize_scale is not None:
        assert resize_size is None, "Choose one mode"
    if resize_size is not None:
        assert resize_scale is None, "Choose one mode"

    img = Image.open(path)

    if resize_size is not None:
        if isinstance(resize_size, int):
            resize_size = resize_size, resize_size
        assert isinstance(resize_size, (list, tuple))
        H, W, *_ = resize_size
        mode = RESIZE_MODES[resize_mode.upper()]
        img = img.resize(size=(W, H), resample=mode)

    if resize_scale is not None:
        if isinstance(resize_scale, (int, float)):
            resize_scale = resize_scale, resize_scale
        assert isinstance(resize_scale, (list, tuple))
        RH, RW, *_ = resize_scale
        W, H = img.size
        mode = RESIZE_MODES[resize_mode.upper()]
        img = img.resize(size=(int(W * RW), int(H * RH)), resample=mode)

    return TF.to_tensor(img) * 255


def to_pil_image(torch_tensor):
    if isinstance(torch_tensor, torch.Tensor):
        torch_tensor = [torch_tensor]

    assert isinstance(torch_tensor, (list, tuple))

    tensors = []
    for t in torch_tensor:
        if t.dim() == 3:
            t = t.unsqueeze(0)
        assert t.dim() == 4
        assert t.size(1) == 3

        t = t.cpu().detach()
        tensors.append(t)

    torch_tensor = torch.cat(tensors, dim=0)
    torch_tensor = torch_tensor.cpu().detach()

    images = []
    observed_formats = []
    for t in torch_tensor:
        if t.mean() >= 1:
            t = t / 255
            images.append(TF.to_pil_image(t))
            observed_formats.append("[0,255]")
            continue

        if t.min() < 0 and t.min() >= -1:
            t = (t + 1) / 2
            images.append(TF.to_pil_image(t))
            observed_formats.append("[-1,1]")
            continue

        images.append(TF.to_pil_image(t))
        observed_formats.append("[0,1]")

    return tuple(images), tuple(observed_formats)
