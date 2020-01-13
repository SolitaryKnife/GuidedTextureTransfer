import torch


def segpatches_to_classidx(segpatches):
    assert isinstance(segpatches, torch.Tensor)
    assert segpatches.dim() == 4

    return segpatches.mean(dim=(2, 3)).argmax(dim=1)


def segmap_to_mask(segmap, classidx):
    assert isinstance(segmap, torch.Tensor)

    if segmap.dim() == 3:
        return (segmap.argmax(dim=0) == classidx).type(torch.float32)

    assert segmap.dim() == 4
    return (segmap.argmax(dim=1) == classidx).type(torch.float32)
