import utils as U
import dataset.cufed5 as c5

from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F

import torchvision.transforms as T
import torchvision.transforms.functional as TF

import images as I

import msoj
import refsr

import metrics as M


pkg = c5.Package("test/raw/CUFED5")

def upscale_bicubic(t):
    if t.dim() == 3:
        t = t.unsqueeze(0)
    assert t.dim() == 4
    assert t.size(0) == 1
    
    return F.interpolate(t, scale_factor=4, mode="bicubic", align_corners=True)
    
is_cuda=True
refsr_model = refsr.get_default_sr_model(cuda=is_cuda)
vgg_model = refsr.get_default_vgg_model(cuda=is_cuda)
sr_model = upscale_bicubic
method = refsr.RefSR(sr_model, refsr_model, vgg_model)


mode = "bicubic"

for r in range(1,6):

    print(f"For similirity level {r}")

    ssim_base = []
    psnr_base = []
    ssim = []
    psnr = []

    for i in tqdm(range(len(pkg))):
        hr = U.data.ImageDataset(f"data/test/proc/CUFED5/{i:03}/hr/*.png", T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: x.cuda().unsqueeze(0) * 255)
        ]))
        lr = U.data.ImageDataset(f"data/test/proc/CUFED5/{i:03}/lr/*.png", T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: x.cuda().unsqueeze(0) * 255)
        ]))
    
        for j in range(len(hr)):
            try:
                x, y = lr[j], hr[j]
                y2 = sr_model(x)

                y, y2 = y / 255, y2 / 255
                
                psnr_base.append(M.pytorch_psnr(y, y2).item())
                ssim_base.append(M.pytorch_ssim(y, y2).item())
            
#             del x, y, y2
            except AssertionError as ae:
                print(f"BASE: There was an assertion error [{i}, {j}]")
                print(ae)
            except Exception as e:
                print(f"BASE: There was an error [{i}, {j}]")
                raise e

        refs = U.data.ImageDataset(f"data/test/proc/CUFED5/{i:03}/s{r}/*.png", T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: x.cuda().unsqueeze(0) * 255)
        ]))
        refs = list(refs)

        for j in range(len(hr)):
            try:
                x, y = lr[j], hr[j]
                y2 = method.upscale_with_ref(x, refs)

                y, y2 = y / 255, y2 / 255
                
                psnr.append(M.pytorch_psnr(y, y2).item())
                ssim.append(M.pytorch_ssim(y, y2).item())

                del x, y, y2
            except AssertionError as ae:
                print(f"NTT: There was an assertion error [{i}, {j}]")
                print(ae)
            except Exception as e:
                print(f"NTT: There was an error [{i}, {j}]")
                raise e

        del refs
            
        print("BASE", sum(ssim_base) / len(ssim_base), sum(psnr_base) / len(psnr_base))
        print("NTT", sum(ssim) / len(ssim), sum(psnr) / len(psnr))

    torch.save(torch.tensor(ssim_base), f"base-{mode}-s{r}-ssim.pth")
    torch.save(torch.tensor(ssim), f"ours-{mode}-s{r}-ssim.pth")
    torch.save(torch.tensor(psnr_base), f"base-{mode}-s{r}-psnr.pth")
    torch.save(torch.tensor(psnr), f"ours-{mode}-s{r}-psnr.pth")
