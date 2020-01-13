import torch
import torch.nn.functional as F

import patches as P
import swapper as X
import segment as S

import architecture as arch


def get_default_sr_model(path="models/all_srntt.pth", eval_mode=True, cuda=True, freeze_param=True):
    model = arch.SRNTT()

    if path is not None:
        model.load_state_dict(torch.load(path))

    if eval_mode:
        model.eval()

    if cuda:
        model.cuda()

    if freeze_param:
        for p in model.parameters():
            p.requires_grad_(False)

    return model


def get_default_vgg_model(path="models/vgg_srntt.pth", eval_mode=True, cuda=True, freeze_param=True):
    model = arch.VGGExtractor()

    if path is not None:
        model.load_state_dict(torch.load(path))

    if eval_mode:
        model.eval()

    if cuda:
        model.cuda()

    if freeze_param:
        for p in model.parameters():
            p.requires_grad_(False)

    return model


class RefSR:

    def __init__(self, sr_model, vgg_model, patch_size=3, stride=1, memsize=512):
        self.model = sr_model
        self.vgg = vgg_model
        self.patch_size = patch_size
        self.stride = stride
        self.memsize = memsize

    def _assert_valid_image(self, x):
        assert isinstance(x, torch.Tensor)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        assert x.dim() == 4
        assert x.size(1) == 3, "Image must be an RGB tensor"
        assert x.mean() >= 1 and x.min() >= 0, "Image must have values in range [0,255]"
        return x

    def _assert_valid_segmap(self, x, nclass=None):
        assert isinstance(x, torch.Tensor)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        assert x.dim() == 4
        if nclass is not None:
            assert x.size(1) == nclass, f"Segmap must have {nclass} channels for the number of classes"
        assert x.mean() > 0 and x.min() >= 0, "Segmap must be a probability map"
        return x

    def build_sr(self, x, map256, map128, map64):
        return self.model(x, refs=[map256, map128, map64])

    def condition_features(self, x):
        return self.vgg(x, "relu3_1")

    def style_features(self, x, layer):
        return self.vgg(x, layer)

    def downscale(self, x):
        x = F.interpolate(x, scale_factor=0.25, mode='bicubic', align_corners=True)
        x[x < 0] = 0
        return x

    def upscale_segmap(self, x, r):
        x = F.interpolate(x, scale_factor=r, mode='bicubic', align_corners=True)
        return x

    def upscale(self, x):
        x = self._assert_valid_image(x)
        return self.model(x)

    def upscale_with_ref(self, x, refs):
        assert isinstance(refs, (list, tuple))

        x = self._assert_valid_image(x)
        refs = [self._assert_valid_image(ref) for ref in refs]

        x_sr = self.upscale(x)
        x_cond = self.condition_features(x_sr)

        del x_sr

        refs_cond = []
        for ref in refs:
            ref_lr = self.downscale(ref)
            ref_sr = self.upscale(ref_lr)
            ref_cond = self.condition_features(ref_sr)
            refs_cond.append(ref_cond)

        del ref, ref_lr, ref_sr, ref_cond

        patches = P.to_patches(refs_cond, patch_size=self.patch_size, stride=self.stride)

        del refs_cond

        maxidx = X.nearest_patch(x_cond, patches, stride=self.stride, memsize=self.memsize)
        _, _, H, W = x_cond.size()

        del patches, x_cond

        features = [None, None, None]
        ratios = [1, 2, 4]
        layers = ["relu3_1", "relu2_1", "relu1_1"]

        for i in range(3):
            ratio = ratios[i]
            layer = layers[i]

            refs_style = []
            for ref in refs:
                ref_style = self.style_features(ref, layer)
                refs_style.append(ref_style)

            del ref_style

            features[i], _ = X.stitch(condition_size=(H, W), ratio=ratios[i], addr=maxidx, refs=refs_style, patch_size=self.patch_size, stride=self.stride)
            if features[i].dim() == 3:
                features[i] = features[i].unsqueeze(0)

            del refs_style

        return self.build_sr(x, *features)

    def upscale_with_ref_with_seg(self, x, x_seg, refs, refs_seg):
        assert isinstance(refs, (list, tuple))
        assert isinstance(refs_seg, (list, tuple))
        assert len(refs) == len(refs_seg)

        x = self._assert_valid_image(x)
        refs = [self._assert_valid_image(ref) for ref in refs]

        x_seg = self._assert_valid_segmap(x_seg)
        nclass = x_seg.size(1)
        refs_seg = [self._assert_valid_segmap(rseg, nclass) for rseg in refs_seg]

        for i in range(len(refs)):
            ref = refs[i]
            ref_seg = refs_seg[i]

            assert ref.size(0) == 1
            assert ref_seg.size(0) == 1
            assert ref.shape[-2:] == ref_seg.shape[-2:]

        x_sr = self.upscale(x)
        x_cond = self.condition_features(x_sr)
        _, _, H, W = x_cond.size()

        del x_sr

        refs_cond = []
        for ref in refs:
            ref_lr = self.downscale(ref)
            ref_sr = self.upscale(ref_lr)
            ref_cond = self.condition_features(ref_sr)
            refs_cond.append(ref_cond)

        del ref, ref_lr, ref_sr, ref_cond

        patches = P.to_patches(refs_cond, patch_size=self.patch_size, stride=self.stride)

        print("patches.size()", patches.size())

        refs_cond_seg = []
        for i in range(len(refs)):
            ref_cond = refs_cond[i]
            ref_seg = refs_seg[i]
            ref_cond_seg = self.downscale(ref_seg)

            assert ref_cond.shape[-2:] == ref_cond_seg.shape[-2:]
            refs_cond_seg.append(ref_cond_seg)

        del refs_cond

        segpatches = P.to_patches(refs_cond_seg, patch_size=self.patch_size, stride=self.stride)
        segpatches_idx = S.segpatches_to_classidx(segpatches)

        del segpatches

        maxidx_per_classidx = [None] * nclass
        for classidx in range(nclass):
            patches_subset = patches[segpatches_idx == classidx]
            if patches_subset.size(0) == 0:
                patches_subset = patches
            maxidx_per_classidx[classidx] = X.nearest_patch(x_cond, patches_subset, stride=self.stride, memsize=self.memsize)

        del x_cond, patches, patches_subset

        features = [None, None, None]
        for classidx in range(nclass):
            maxidx = maxidx_per_classidx[classidx]

            print()
            print(f"Doing classidx {classidx:02}")

            ratios = [1, 2, 4]
            layers = ["relu3_1", "relu2_1", "relu1_1"]

            for i in range(3):
                print()
                print(f"Doing layer {i + 1}")
                ratio = ratios[i]
                layer = layers[i]

                mask_seg = self.upscale_segmap(x_seg, ratio)
                mask = S.segmap_to_mask(mask_seg, classidx).unsqueeze(0)
                if mask.mean() == 0:
                    print(f"Class {classidx} for size {ratio} is skipped")
                    continue

                refs_style = []
                for ref in refs:
                    ref_style = self.style_features(ref, layer)
                    refs_style.append(ref_style)

                del ref_style

                def filter_patches(patches):
                    subset = patches[segpatches_idx == classidx]
                    if subset.size(0) == 0:
                        return patches
                    return subset

                feature = X.stitch(
                    condition_size=(H, W),
                    ratio=ratios[i],
                    addr=maxidx,
                    refs=refs_style,
                    patch_size=self.patch_size,
                    stride=self.stride,
                    patch_postprocess=filter_patches)[0]

                if feature.dim() == 3:
                    feature = feature.unsqueeze(0)

                print("mask.mean()", mask.mean())
                print("mask.size()", mask.size())

                if features[i] is None:
                    features[i] = feature * mask
                else:
                    features[i] += feature * mask

                del refs_style, mask, feature

        return self.build_sr(x, *features)
