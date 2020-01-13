import torch
import torchvision

from PIL import Image

import architecture as arch
import swapper as swap

assert torch.cuda.is_available(), "CUDA is required"

input_image_path = "samples/000_0.png"
input_seg_path = "samples/000_0_bic.pth"

refs_image_path = [
    "samples/000_1.png",
    "samples/000_2.png",
    "samples/000_3.png",
    "samples/000_4.png",
    "samples/000_5.png",
]
refs_seg_path = [
    "samples/000_1_bic.pth",
    "samples/000_2_bic.pth",
    "samples/000_3_bic.pth",
    "samples/000_4_bic.pth",
    "samples/000_5_bic.pth",
]

assert len(refs_image_path) == len(refs_seg_path)


def load_image(path):
    return torchvision.transforms.functional.to_tensor(Image.open(path)) * 255


def load_segmap(path):
    return torch.load(path)


def match_image_and_segmentation(img, segmap):
    assert img.dim() == 3
    assert segmap.dim() == 3

    _, H_img, W_img = img.size()
    _, H_seg, W_seg = segmap.size()

    H, W = min(H_img, H_seg), min(W_img, W_seg)

    return img[:, :H, :W], segmap[:, :H, :W]


x_hr = load_image(input_image_path)
x_seg = load_segmap(input_seg_path)
x_hr, x_seg = match_image_and_segmentation(x_hr, x_seg)

refs_hr = []
refs_seg = []
for i in range(len(refs_image_path)):
    ref_image_path = refs_image_path[i]
    ref_seg_path = refs_seg_path[i]

    ref_hr = load_image(ref_image_path)
    ref_seg = load_segmap(ref_seg_path)

    ref_hr, ref_seg = match_image_and_segmentation(ref_hr, ref_seg)

srntt_py = arch.SRNTT()
srntt_py.load_state_dict(torch.load("models/all_srntt.pth"))
srntt_py.cuda()
srntt_py.eval()

for p in srntt_py.parameters():
    p.requires_grad_(False)


vgg_py = arch.VGGExtractor()
vgg_py.load_state_dict(torch.load("models/vgg_srntt.pth"))
vgg_py.cuda()
vgg_py.eval()

for p in vgg_py.parameters():
    p.requires_grad_(False)

print(x_hr.size())
print(x_seg.mean())
