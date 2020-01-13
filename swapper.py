import torch
import torch.nn as nn
import torch.nn.functional as F

import patches as P


def nearest_patch_0(content, patches, *, stride=1):
    assert isinstance(content, torch.Tensor)
    assert isinstance(patches, torch.Tensor)

    if content.dim() == 3:
        content = content.unsqueeze(0)

    assert content.size(0) == 1
    assert content.dim() == 4
    assert patches.dim() == 4

    norm = (patches ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
    patch_norm = patches / norm

    corr = F.conv2d(content, patch_norm, stride=stride).squeeze(0)
    argmax = corr.argmax(dim=0)

    return argmax


def nearest_patch(content, patches, *, stride=1, memsize=512):
    assert isinstance(content, torch.Tensor)
    assert isinstance(patches, torch.Tensor)

    if content.dim() == 3:
        content = content.unsqueeze(0)

    assert content.size(0) == 1
    assert content.dim() == 4
    assert patches.dim() == 4

    _, C, H, W = content.size()
    N = patches.size(0)

    batch_size = int(1024 ** 2 * memsize / (H * W))

    if batch_size >= N:
        return nearest_patch_0(content, patches, stride=stride)

    norm = (patches ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
    patch_norm = patches / norm

    final_argmax, final_max = None, None
    for idx in range(0, N, batch_size):

        idx_end = min(idx + batch_size, N)
        patch_batch = patch_norm[idx:idx_end]

        corr = F.conv2d(content, patch_batch, stride=stride)
        corr = corr.squeeze(0)

        del patch_batch

        temp_argmax = corr.argmax(dim=0) + idx
        temp_max, _ = corr.max(dim=0)

        if final_argmax is None:
            final_argmax = temp_argmax
            final_max = temp_max
        else:
            indeces = temp_max > final_max
            final_max[indeces] = temp_max[indeces]
            final_argmax[indeces] = temp_argmax[indeces]

    assert final_argmax is not None
    return final_argmax


def stitch(condition_size, ratio, addr, refs, patch_size=3, stride=1, patch_postprocess=None):
    if isinstance(refs, torch.Tensor):
        refs = [refs]
    assert isinstance(refs, (list, tuple))

    for ref in refs:
        assert isinstance(ref, torch.Tensor)
        assert ref.dim() == 3 or ref.dim() == 4

    if isinstance(condition_size, int):
        condition_size = condition_size, condition_size
    assert isinstance(condition_size, (tuple, list))
    H, W, *_ = condition_size

    print(f"CondH={H}, CondW={W}")

    if isinstance(ratio, int):
        ratio = ratio, ratio
    assert isinstance(ratio, (tuple, list))
    RH, RW, *_ = ratio

    print(f"RH={RH}, RW={RW}")

    if isinstance(patch_size, int):
        patch_size = patch_size, patch_size
    assert isinstance(patch_size, (tuple, list))
    PH, PW, *_ = patch_size
    print(f"Patch_H={PH}, Patch_W={PW}")

    if isinstance(stride, int):
        stride = stride, stride
    assert isinstance(stride, (tuple, list))
    SH, SW, *_ = stride
    print(f"Stride_H={SH}, Stride_W={SW}")

    assert isinstance(addr, torch.Tensor)
    assert addr.dim() == 2
    device = addr.get_device()

    AH, AW = addr.size()
    print(f"Addr_H={AH}, Addr_W={AW}")

    assert patch_postprocess is None or callable(patch_postprocess)

    patches = P.to_patches(refs, patch_size=(PH * RH, PW * RW), stride=(SH * RH, SW * RW))
    patches = patches.cpu() if device < 0 else patches.to(device=device)
    if patch_postprocess is not None:
        patches = patch_postprocess(patches)
        print("Subset", patches.size())

    print("PatchSize", patches.size())

    _, C, PH, PW = patches.size()

    print(f"NewPatch_H={PH}, NewPatch_W={PW}")

    outputs = torch.zeros(C, H * RH, W * RH)
    outputs = outputs.cpu() if device < 0 else outputs.to(device=device)

    counts = torch.zeros(H * RH, W * RH)
    counts = counts.cpu() if device < 0 else counts.to(device=device)

    print("outputs.size", outputs.size())
    print("counts.size", counts.size())

    for i in range(AH):

        i_start = i * RH
        i_end = i * RH + PH

        for j in range(AW):

            j_start = j * RW
            j_end = j * RW + PW

            try:
                patch = patches[addr[i, j], ...]
                outputs[:, i_start:i_end, j_start:j_end] += patch
                counts[i_start:i_end, j_start:j_end] += 1.0
            except Exception as e:
                print(f"i={i}, j={j}")
                print("addr[i, j]", addr[i, j])
                print(f"range_i={range(i_start, i_end)}")
                print(f"range_j={range(j_start, j_end)}")
                print("outputs[:, i_start:i_end, j_start:j_end].size()", outputs[:, i_start:i_end, j_start:j_end].size())
                # print("patch.size()", patch.size())
                raise e

    print(f"last_i={i}, last_i_start={i_start}, last_i_end={i_end}")
    print(f"last_j={j}, last_j_start={j_start}, last_j_end={j_end}")
    return outputs / counts.view(1, H * RH, W * RH), counts


# def to_whole(p, output_size, dilation=1, padding=0, stride=1):
#     assert p.dim() == 4
#     N, C, H, W = p.size()
#     assert H == W
#     patch_size = H

#     x = p.view(N, -1)
#     x = x.permute(1, 0)
#     x = x.unsqueeze(0)

#     divisor = F.fold(
#         torch.ones_like(x),
#         output_size=output_size,
#         kernel_size=patch_size,
#         dilation=dilation,
#         padding=padding,
#         stride=stride)
#     numerator = F.fold(
#         x,
#         output_size=output_size,
#         kernel_size=patch_size,
#         dilation=dilation,
#         padding=padding,
#         stride=stride)

#     return (numerator / divisor).squeeze(0)


def patch_similarity(p1, p2):
    assert p1.dim() == 4
    assert p2.dim() == 4
    assert p1.shape[1:] == p2.shape[1:]

    norm = (p2 ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
    p2_norm = p2 / norm

    corr = F.conv2d(p1, p2_norm).squeeze(0)

    return corr.argmax(dim=1).view(-1)


def patch_similarity_memsave(p1, p2, memsize=512):
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
