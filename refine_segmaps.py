import sys
import os
import os.path as path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import numpy as np

from PIL import Image as I
from glob import glob

def prepare_folder(p):
    dirname = path.dirname(p)
    try:
        os.makedirs(dirname)
    except:
        pass

def pairwise_distance(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2).sqrt()

centroids = []
for r in [0, 1]:
    for g in [0, 1]:
        for b in [0, 1]:
            centroids.append([r, g, b])
centroids = torch.tensor(np.array(centroids), dtype=torch.float32)

NC, C = centroids.size()

print("Color centroids are:")
print(f"_\tR\tG\tB")
temp = (centroids * 255).type(torch.int16)
for i in range(NC):
    print(f"{i}\t{temp[i,0]}\t{temp[i,1]}\t{temp[i,2]}")

print()

filepattern = sys.argv[1]
print(f"Finding files of pattern: {filepattern}")

files = glob(filepattern)
print(f"Found {len(files)} files!")

print()

for i, filepath in enumerate(files):
    print(f"Processing ({i + 1}/{len(files)}): {filepath}")

    
    dirname = path.dirname(filepath)
    filename = path.basename(filepath)
    basename, ext = path.splitext(filename)

    img = I.open(filepath)
    tensor = TF.to_tensor(img)
    C, H, W = tensor.size()
    print(f"Image size is {H} x {W}")

    tensor = tensor.view(C, H * W).permute(1, 0)
    dists = pairwise_distance(centroids, tensor)
    nearest = 1 - F.softmax(dists, dim=0)
    probmap = nearest.view(NC, H, W)

    probmap_save = path.join(dirname, "prob", basename) + ".pth"
    prepare_folder(probmap_save)
    torch.save(probmap, probmap_save)
    print(f"probablity_map: {probmap_save}")

    maxclass = probmap.argmax(dim=0)
    refined_colormap = centroids[maxclass, :].permute(2, 0, 1)

    colormap_save = path.join(dirname, "refined", basename) + ".png"
    prepare_folder(colormap_save)
    TF.to_pil_image(refined_colormap).save(colormap_save)

    print()

# print(refined_colormap.size())