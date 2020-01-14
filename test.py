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


is_cuda=True
refsr_model = refsr.get_default_sr_model(cuda=is_cuda)
vgg_model = refsr.get_default_vgg_model(cuda=is_cuda)
method = refsr.RefSR(refsr_model, refsr_model, vgg_model)

model_names = ["edsr", "srgan"]
sim_levels = 5

ssim_data = {model_name: [[] for _ in range(sim_levels)] for model_name in model_names}
psnr_data = {model_name: [[] for _ in range(sim_levels)] for model_name in model_names}

base_ssim_data = {model_name: [[] for _ in range(sim_levels)] for model_name in model_names}
base_psnr_data = {model_name: [[] for _ in range(sim_levels)] for model_name in model_names}

transform = T.Compose([
    T.ToTensor(),
    T.Lambda(lambda x: x.cuda().unsqueeze(0) * 255)
])

for sim_level in range(1,6):
    for i in tqdm(range(126)):
        hrlist = U.data.ImageDataset(f"data/test/proc/cufed5/{i:03}/hr/*.png", transform)
        lrlist = U.data.ImageDataset(f"data/test/proc/cufed5/{i:03}/lr/*.png", transform)
        reflist = list(U.data.ImageDataset(f"data/test/proc/cufed5/{i:03}/s{sim_level}/*.png", transform))
        
        models = {
            "edsr": U.data.ImageDataset(f"data/test/proc/cufed5/{i:03}/__edsr/*.png", transform),
            "srgan": U.data.ImageDataset(f"data/test/proc/cufed5/{i:03}/__srgan/*.png", transform),
        }

        for model_name in models:
            model_outputs = models[model_name]
            
            for idx in range(len(model_outputs)):
                x = model_outputs[idx]
                y = hrlist[idx]

                x, y = x / 255, y / 255

                base_psnr = M.pytorch_psnr(x, y).item()
                base_ssim = M.pytorch_ssim(x, y).item()

                base_psnr_data[model_name][sim_level - 1].append(base_psnr)
                base_ssim_data[model_name][sim_level - 1].append(base_ssim)

            for idx in range(len(model_outputs)):
                x = model_outputs[idx]
                y = method.upscale_with_ref(
                    x=x,
                    refs=reflist)

                y_small = F.interpolate(y, scale_factor=0.25, mode="bicubic", align_corners=True)
                y2 = hrlist[idx]

                y_small, y2 = y_small / 255, y2 / 255
                psnr = M.pytorch_psnr(y_small, y2).item()
                ssim = M.pytorch_ssim(y_small, y2).item()
                
                base_psnr_data[model_name][sim_level - 1].append(psnr)
                base_ssim_data[model_name][sim_level - 1].append(ssim)

import json
with open('ssim_data.json', 'w') as f:
    json.dump(ssim_data, f)
with open('psnr_data.json', 'w') as f:
    json.dump(ssim_data, f)
with open('base_ssim_data.json', 'w') as f:
    json.dump(ssim_data, f)
with open('base_psnr_data.json', 'w') as f:
    json.dump(ssim_data, f)