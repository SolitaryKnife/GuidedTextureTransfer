import torch
from PIL import Image
import dataset.urban100 as u100
from tqdm import tqdm

import utils as U


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

pkg = u100.Package("data/test/raw/Urban100")
cache_dir = "data/test/proc/Urban100"
size = 120


def preprocess(img, size):
    assert isinstance(size, int)
    W, H = img.size
    assert size <= H and size <= W

    xH, xW = H % size, W % size
    for i in range(0, H - xH, size):
        for j in range(0, W - xW, size):
            top = i
            left = j
            bottom = i + size
            right = j + size
            patch = img.crop((left, top, right, bottom))
            yield patch


for i, (img_lr, img_hr) in enumerate(tqdm(pkg.image_dataset())):
    W_hr, H_hr = img_hr.size
    W_lr, H_lr = img_lr.size

    ratio = W_hr // W_lr
    assert ratio == W_hr / W_lr
    assert ratio == H_hr / H_lr

    lr_size = size // ratio
    assert lr_size == size / ratio

    def build_patches(img, size, dirname):
        j = 0
        for patch in preprocess(img, size):
            patch.save(U.io.prepare_path(cache_dir, f"{i:03}", dirname, f"{j:03}.png"))
            j += 1
        return j

    count1 = build_patches(img_lr, lr_size, "lr")
    count2 = build_patches(img_hr, size, "hr")

    assert count1 == count2
