import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models.vgg as vgg


class PixelShuffleTensorflow(nn.Module):

    def __init__(self, upscale_factor):
        super().__init__()

        self.upscale_factor = upscale_factor

    def forward(self, x):
        block_size = self.upscale_factor

        x = x.permute(0, 2, 3, 1)

        batch, height, width, depth = x.size()
        new_depth = depth // (block_size ** 2)
        x = x.reshape(batch, height, width, block_size, block_size, new_depth)

        x = x.permute(0, 1, 3, 2, 4, 5)

        new_height = height * block_size
        new_width = width * block_size
        x = x.reshape(batch, new_height, new_width, -1)

        x = x.permute(0, 3, 1, 2)
        return x


class LinearShift(nn.Module):

    def __init__(self, m_shape=[1], b_shape=[1]):
        super().__init__()

        self.m = nn.Parameter(torch.ones(*m_shape, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(*b_shape, dtype=torch.float32))

    def forward(self, x):
        return self.m * x + self.b

    def __repr__(self):
        return (self.__class__.__name__ + "(m={}, b={})").format(
            self.m.clone().detach().numpy(),
            self.b.clone().detach().numpy())


class ResBlock(nn.Module):
    def __init__(self, channels=64, ksize=3, batchnorm=True):
        super().__init__()

        padding = ksize // 2

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=ksize, padding=padding),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=ksize, padding=padding),
            nn.BatchNorm2d(num_features=channels),
        )

    def forward(self, x):
        y = self.model(x)
        return x + y


class ContentFeature(nn.Module):

    def __init__(self, in_channels=3, body_channels=64, out_channels=64, ksize=3, resblocks=16):
        super().__init__()

        padding = ksize // 2

        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=body_channels, kernel_size=ksize, padding=padding),
            nn.ReLU(inplace=True)
        )

        layers = [ResBlock(channels=body_channels, ksize=ksize) for i in range(resblocks)]
        layers += [
            nn.Conv2d(in_channels=body_channels, out_channels=out_channels, kernel_size=ksize, padding=padding),
            nn.BatchNorm2d(num_features=out_channels)
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        y = self.pre(x)
        return self.body(y) + y


class NetUpscale(nn.Module):

    def __init__(self, channels, ksize=3, factor=2, padtype=None):
        super().__init__()

        padding = ksize // 2
        out_channels = channels * (factor ** 2)

        self.conv = nn.Conv2d(in_channels=channels, out_channels=out_channels, kernel_size=ksize, padding=padding)
        self.shuffle = PixelShuffleTensorflow(factor)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.shuffle(self.conv(x)))


class BasicSR(nn.Module):

    def __init__(self, in_channels=64, out_channels=3, ksize=3, upscale_steps=2, upscale_factor=2):
        super().__init__()

        layers = [NetUpscale(channels=in_channels, ksize=ksize, factor=upscale_factor) for _ in range(upscale_steps)]
        layers += [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def __getitem__(self, idx):
        return self.model[idx]

    def __len__(self):
        return len(self.model)


class TextureFusion(nn.Module):

    def __init__(self, in_channels, ref_channels, out_channels, ksize=3, resblocks=16):
        super().__init__()

        padding = ksize // 2

        layers = [
            nn.Conv2d(in_channels=in_channels + ref_channels, out_channels=out_channels, kernel_size=ksize, padding=padding),
            nn.ReLU(inplace=True)
        ]
        layers += [ResBlock(channels=out_channels, ksize=ksize) for i in range(resblocks)]
        layers += [
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=ksize, padding=padding),
            nn.BatchNorm2d(num_features=out_channels)
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x_in, x_ref):
        x = torch.cat([x_in, x_ref], dim=1)
        y = self.model(x)
        return y + x_in

    def __getitem__(self, idx):
        return self.model[idx]

    def __len__(self):
        return len(self.model)


class TextureTransferSR(nn.Module):

    def __init__(self, ksize=3, resblocks=16):
        super().__init__()

        padding = ksize // 2

        self.weight_norm_1 = nn.Sequential(LinearShift(), nn.Sigmoid())
        self.texture_fusion_1 = TextureFusion(in_channels=64, ref_channels=256, out_channels=64, ksize=ksize, resblocks=resblocks)
        self.net_upscale_1 = NetUpscale(64, ksize=ksize, factor=2)

        self.weight_norm_2 = nn.Sequential(LinearShift(), nn.Sigmoid())
        self.texture_fusion_2 = TextureFusion(in_channels=64, ref_channels=128, out_channels=64, ksize=ksize, resblocks=resblocks // 2)
        self.net_upscale_2 = NetUpscale(64, ksize=ksize, factor=2)

        self.weight_norm_3 = nn.Sequential(LinearShift(), nn.Sigmoid())
        self.texture_fusion_3 = TextureFusion(in_channels=64, ref_channels=64, out_channels=64, ksize=ksize, resblocks=resblocks // 4)

        self.final = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=ksize, padding=padding),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1),
            nn.Tanh()
        )

    @staticmethod
    def weight_scale(w, scale):
        return F.interpolate(w, scale_factor=scale)

    def forward(self, x, ref1, ref2, ref3, weights=None):
        y = x

        if weights is not None:
            ref1 = ref1 * self.weight_norm_1(weights)
        y = self.texture_fusion_1(y, ref1)
        y = self.net_upscale_1(y)

        if weights is not None:
            w = self.__class__.weight_scale(weights, 2)
            ref2 = ref2 * self.weight_norm_2(w)
        y = self.texture_fusion_2(y, ref2)
        y = self.net_upscale_2(y)

        if weights is not None:
            w = self.__class__.weight_scale(weights, 4)
            ref3 = ref3 * self.weight_norm_3(w)
        y = self.texture_fusion_3(y, ref3)

        y = self.final(y)
        return y


class SRNTT(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = ContentFeature()
        self.sr_basic = BasicSR()
        self.sr_guided = TextureTransferSR()

    def preprocess(self, x):
        return x / 127.5 - 1

    def postprocess(self, x):
        return (x + 1) * 127.5

    def forward(self, x, refs=None, weights=None):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        assert x.dim() == 4
        assert x.mean() >= 1, "x must have values [0-255]"

        x = self.preprocess(x)

        y = self.features(x)

        if refs is None:
            y = self.sr_basic(y)
            return self.postprocess(y)

        assert isinstance(refs, (list, tuple)), "Refs must be a list or a tuple"
        assert len(refs) == 3, "Refs must have length 3"

        assert refs[0].dim() == 4
        assert refs[1].dim() == 4
        assert refs[2].dim() == 4

        assert refs[0].size(1) == 256
        assert refs[1].size(1) == 128
        assert refs[2].size(1) == 64

        _, _, H1, W1 = refs[0].size()
        _, _, H2, W2 = refs[1].size()
        _, _, H3, W3 = refs[2].size()

        assert H1 == H3 / 4 and W1 == W3 / 4
        assert H2 == H3 / 2 and W2 == W3 / 2

        ref1, ref2, ref3, *_ = refs
        y = self.sr_guided(y, ref1, ref2, ref3, weights)

        return self.postprocess(y)


def vgg_preprocess(x):
    try:
        return x - torch.tensor([123.68, 116.779, 103.939]).view(1, -1, 1, 1)
    except:
        return x - torch.tensor([123.68, 116.779, 103.939]).view(1, -1, 1, 1).cuda()


class VGGExtractor(nn.Module):

    @staticmethod
    def layername_index_mapping(features):
        mapping = {}

        n, m = 1, 0
        for i, layer in enumerate(features):
            if isinstance(layer, nn.Conv2d):
                m += 1
                mapping["conv{}_{}".format(n, m)] = i
            elif isinstance(layer, nn.ReLU):
                mapping["relu{}_{}".format(n, m)] = i
            elif isinstance(layer, nn.BatchNorm2d):
                mapping["batch{}_{}".format(n, m)] = i
            elif isinstance(layer, nn.MaxPool2d):
                mapping["pool{}".format(n)] = i
                n += 1
                m = 0

        return mapping

    @staticmethod
    def fetches_to_idxs(fetches, mapping):
        idxs = []
        for idx in fetches:
            if isinstance(idx, int):
                idxs.append(idx)
            elif isinstance(idx, str):
                try:
                    idx = mapping[idx]
                except:
                    raise ValueError("Layer `{}` not found".format(idx))
                idxs.append(idx)
            else:
                raise ValueError("Expected `fetches` to be list[int], list[str]")
        return idxs

    def __init__(self, vgg_nlayer=19, vgg_bn=False, preprocess=vgg_preprocess, requires_grad=False):
        super().__init__()

        model_name = "vgg{}{}".format(vgg_nlayer, "_bn" if vgg_bn else "")
        self.model = getattr(vgg, model_name)().features
        self.mapping = self.__class__.layername_index_mapping(self.model)

        self.preprocess = preprocess
        if self.preprocess is None:
            self.preprocess = lambda x: x

        for p in self.model.parameters():
            p.requires_grad_(requires_grad)
        for m in self.model:
            if isinstance(m, nn.ReLU):
                m.inplace = False

    def forward(self, x, fetches=None):
        if fetches is None:
            return self.model(x)
        if isinstance(fetches, (int, str)):
            fetches = [fetches]
            fetches = self.__class__.fetches_to_idxs(fetches, self.mapping)
            return self.forward0(x, fetches)[0]
        if isinstance(fetches, (tuple, list)):
            fetches = self.__class__.fetches_to_idxs(fetches, self.mapping)
            return self.forward0(x, fetches)
        raise ValueError("Expected `fetches` to be int, str, list[int], list[str]")

    def forward0(self, x, idxs):
        outputs = {}

        if None in idxs:
            outputs[None] = x

        y = self.preprocess(x)

        if -1 in idxs:
            outputs[-1] = y

        last_idxs = max(idxs)

        for idx, layer in enumerate(self.model):
            y = layer(y)

            if idx in idxs:
                outputs[idx] = y

            if idx > last_idxs:
                break

        return [outputs[idx] for idx in idxs]
