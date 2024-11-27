"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models

from ..util import get_ckpt_path


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "talking_head/vae/models/modules/autoencoding/lpips/loss")
        self.load_state_dict(
            torch.load(ckpt, map_location=torch.device("cpu")), strict=False
        )
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

        # Traceback (most recent call last):
        #   File "/mnt/workspace/yifanzhang/talking-head/train_vae.py", line 662, in <module>
        #     model = instantiate_from_config(config.model)
        #   File "/mnt/workspace/yifanzhang/talking-head/talking_head/models/vae/util.py", line 188, in instantiate_from_config
        #     return get_obj_from_str(config["target"])(**config.get("params", dict()))
        #   File "/mnt/workspace/yifanzhang/talking-head/talking_head/models/vae/models/autoencoder.py", line 137, in __init__
        #     self.loss: torch.nn.Module = instantiate_from_config(loss_config)
        #   File "/mnt/workspace/yifanzhang/talking-head/talking_head/models/vae/util.py", line 188, in instantiate_from_config
        #     return get_obj_from_str(config["target"])(**config.get("params", dict()))
        #   File "/mnt/workspace/yifanzhang/talking-head/talking_head/models/vae/modules/autoencoding/losses/discriminator_loss.py", line 45, in __init__
        #     self.perceptual_loss = LPIPS().eval()
        #   File "/mnt/workspace/yifanzhang/talking-head/talking_head/models/vae/modules/autoencoding/lpips/loss/lpips.py", line 24, in __init__
        #     self.load_from_pretrained()
        #   File "/mnt/workspace/yifanzhang/talking-head/talking_head/models/vae/modules/autoencoding/lpips/loss/lpips.py", line 30, in load_from_pretrained
        #     self.load_state_dict(
        #   File "/opt/conda/envs/hipa/lib/python3.10/site-packages/torch/nn/modules/module.py", line 2152, in load_state_dict
        #     raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
        # RuntimeError: Error(s) in loading state_dict for LPIPS:
        # 	Missing key(s) in state_dict: "scaling_layer.shift", "scaling_layer.scale", "net.slice1.0.weight", "net.slice1.0.bias", "net.slice1.2.weight", "net.slice1.2.bias", "net.slice2.5.weight", "net.slice2.5.bias", "net.slice2.7.weight", "net.slice2.7.bias", "net.slice3.10.weight", "net.slice3.10.bias", "net.slice3.12.weight", "net.slice3.12.bias", "net.slice3.14.weight", "net.slice3.14.bias", "net.slice4.17.weight", "net.slice4.17.bias", "net.slice4.19.weight", "net.slice4.19.bias", "net.slice4.21.weight", "net.slice4.21.bias", "net.slice5.24.weight", "net.slice5.24.bias", "net.slice5.26.weight", "net.slice5.26.bias", "net.slice5.28.weight", "net.slice5.28.bias". 

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.load_state_dict(
            torch.load(ckpt, map_location=torch.device("cpu")), strict=False
        )
        return model

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(
                outs1[kk]
            )
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [
            spatial_average(lins[kk].model(diffs[kk]), keepdim=True)
            for kk in range(len(self.chns))
        ]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer(
            "shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )
        self.register_buffer(
            "scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = (
            [
                nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False 
    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x, eps=1e-10): 
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)
