import torch
from PIL import Image
import dataset.cufed5 as c5
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

pkg = c5.Package("data/test/raw/CUFED5")
cache_dir = "data/test/proc/CUFED5"
size = 120
lr_size = 30
lr_mode = RESIZE_MODES["BICUBIC"]


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


for i, (img_hr, ref1, ref2, ref3, ref4, ref5) in enumerate(tqdm(pkg.image_dataset())):

    j = 0
    for patch in preprocess(img_hr, size):
        patch.save(U.io.prepare_path(cache_dir, f"{i:03}", "hr", f"{j:03}.png"))

        patch = patch.resize(size=(lr_size, lr_size), resample=lr_mode)
        patch.save(U.io.prepare_path(cache_dir, f"{i:03}", "lr", f"{j:03}.png"))
        j += 1

    def build_ref(img, dirname):
        j = 0
        for patch in preprocess(img, size):
            patch.save(U.io.prepare_path(cache_dir, f"{i:03}", dirname, f"{j:03}.png"))
            j += 1

    build_ref(ref1, "s1")
    build_ref(ref2, "s2")
    build_ref(ref3, "s3")
    build_ref(ref4, "s4")
    build_ref(ref5, "s5")
