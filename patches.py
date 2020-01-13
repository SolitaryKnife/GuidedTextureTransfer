import torch
import torch.nn.functional as F


def to_patches(x, *, patch_size=3, dilation=1, padding=0, stride=1):
    if isinstance(x, (list, tuple)):
        # batches = []
        # for t in x:
        #     p = to_patches(t, patch_size=patch_size, dilation=dilation, padding=padding, stride=stride)
        #     batches.append(p)
        # return torch.cat(batches, dim=0)
        batches = None
        for t in x:
            p = to_patches(t, patch_size=patch_size, dilation=dilation, padding=padding, stride=stride)
            if batches is None:
                batches = p
            else:
                batches = torch.cat([batches, p], dim=0)
        return batches

    assert isinstance(x, torch.Tensor)

    if x.dim() == 4:
        return to_patches(list(x), patch_size=patch_size, dilation=dilation, padding=padding, stride=stride)

    assert x.dim() == 3
    x = x.unsqueeze(0)

    if isinstance(patch_size, int):
        patch_size = patch_size, patch_size
    assert isinstance(patch_size, (tuple, list))
    PH, PW, *_ = patch_size

    p = F.unfold(x, kernel_size=patch_size, dilation=dilation, padding=padding, stride=stride)
    p = p.squeeze(0)
    p = p.permute(1, 0)

    _, C, _, _ = x.size()
    N, _ = p.size()
    p = p.view(N, C, PH, PW)

    return p


def nearest_patch(p1, p2):
    assert p1.dim() == 4
    assert p2.dim() == 4
    assert p1.shape[1:] == p2.shape[1:]

    norm = (p2 ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
    p2_norm = p2 / norm

    corr = F.conv2d(p1, p2_norm).squeeze(0)

    return corr.argmax(dim=1).view(-1)


def nearest_patch_memsave(p1, p2, *, memsize=512):
    assert p1.dim() == 4
    assert p2.dim() == 4
    assert p1.shape[1:] == p2.shape[1:]

    norm = (p2 ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
    p2_norm = p2 / norm

    N, C, H, W = p1.size()
    batch_size = int(1024 ** 2 * memsize / (C * H * W) ** 2)

    outputs = []
    for idx in range(0, N, batch_size):
        idx_end = min(idx + batch_size, N)
        p = p1[idx:idx_end]
        corr = F.conv2d(p, p2_norm).squeeze(-1).squeeze(-1)
        argmax = corr.argmax(dim=1).view(-1)
        outputs.append(argmax)

    return torch.cat(outputs)
