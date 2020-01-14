import utils as U
import dataset.cufed5 as c5

from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F

import torchvision.transforms as T
import torchvision.transforms.functional as TF

import images as I

import refsr

import metrics as M
import patches as P

import os


is_cuda=True
refsr_model = refsr.get_default_sr_model(cuda=is_cuda)
vgg_model = refsr.get_default_vgg_model(cuda=is_cuda)
method = refsr.RefSR(refsr_model, refsr_model, vgg_model)

transform = T.Compose([
    T.ToTensor(),
    T.Lambda(lambda x: x.cuda().unsqueeze(0) * 255)
])

for method in [bicubic, edsr, srgan]:

    print(method)
    
    try:
        os.makedirs(f"cufed5_{method}_ours/big/")
        os.makedirs(f"cufed5_{method}_ours/small/")
    except:
        pass

    for i, img in enumerate(tqdm(U.data.ImageDataset(f"cufed5_{method}/**.png", transform))):
        refs = U.data.ImageDataset(f"data/test/raw/CUFED5/{i:03}_*.png", transform)
        assert len(refs) == 5

        y = method.upscale_with_ref(img, refs).cpu().squeeze(0)


        img_big = TF.to_pil_image(y / 255)
        img_small = img_big.resize((img_big.size[0] // 4, img_big.size[1] // 4), Image.BICUBIC)

        path1 = f"cufed5_{method}_ours/big/{i:03}_0.png"
        img_big.save(path1)

        path2 = f"cufed5_{method}_ours/small/{i:03}_0.png"
        img_small.save(path2)