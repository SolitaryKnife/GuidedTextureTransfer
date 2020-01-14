import torch.nn.functional as _F
import torch as _torch

import lib as _lib

ssim_pytorch_lib = _lib.ssim


def pytorch_ssim(img1, img2, window_size=11, size_average=True):
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    ssim = ssim_pytorch_lib.ssim(img1, img2, window_size=window_size, size_average=size_average)
    return ssim


def pytorch_psnr(img1, img2, maxval=1.0):
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    mse = _F.mse_loss(img1, img2)
    constant = 20 * _torch.log10(_torch.tensor(maxval))
    psnr = constant - 10 * _torch.log10(mse)
    return psnr