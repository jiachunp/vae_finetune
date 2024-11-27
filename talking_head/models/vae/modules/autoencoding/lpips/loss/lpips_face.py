"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models

from facenet_pytorch import InceptionResnetV1
from ..util import get_ckpt_path


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [256, 896, 1792, 1792]  # Adjusted to match new slices
        self.net = InceptionResnetV1_Features(pretrained=True, ckpt_path='/mnt/data/yifanzhang/talking-head/talking_head/vae/models/modules/autoencoding/lpips/loss/vggface.pt').eval()
        
        for param in self.net.parameters():
            param.requires_grad = False

        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)

        for param in self.lin0.parameters():
            param.requires_grad = True
        for param in self.lin1.parameters():
            param.requires_grad = True
        for param in self.lin2.parameters():
            param.requires_grad = True
        for param in self.lin3.parameters():
            param.requires_grad = True

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk](diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.Tensor([0.5, 0.5, 0.5])[None, :, None, None]
        self.scale = torch.Tensor([0.5, 0.5, 0.5])[None, :, None, None]

    def forward(self, x):
        return (x - self.shift.to(x.device)) / self.scale.to(x.device)

class NetLinLayer(nn.Module):
    def __init__(self, chn_in, chn_out=1, use_dropout=True):
        super().__init__()
        layers = [nn.Dropout(), nn.Conv2d(chn_in, chn_out, 1, bias=False)]
        self.model = nn.Sequential(*layers)
        self.initialize_weights()

    def initialize_weights(self):
        # Custom initialization to ensure positive weights
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight, a=0.01, b=1.0)  # Initialize with uniform distribution with positive values

    def forward(self, x):
        return self.model(x)

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


class InceptionResnetV1_Features(nn.Module):
    def __init__(self, pretrained=True, ckpt_path=None):
        super(InceptionResnetV1_Features, self).__init__()
        self.inception_resnet = InceptionResnetV1(pretrained='vggface2' if pretrained else None, ckpt_path=ckpt_path).eval()
        
        if ckpt_path:
            state_dict = torch.load(ckpt_path)
            self.inception_resnet.load_state_dict(state_dict)

        # Define the layers based on the provided architecture
        self.slice1 = nn.Sequential(
            self.inception_resnet.conv2d_1a,
            self.inception_resnet.conv2d_2a,
            self.inception_resnet.conv2d_2b,
            self.inception_resnet.maxpool_3a,
            self.inception_resnet.conv2d_3b,
            self.inception_resnet.conv2d_4a,
            self.inception_resnet.conv2d_4b
        )  # Output channels: 256

        self.slice2 = nn.Sequential(
            self.inception_resnet.repeat_1,
            self.inception_resnet.mixed_6a
        )  # Output channels: 896

        self.slice3 = nn.Sequential(
            self.inception_resnet.repeat_2,
            self.inception_resnet.mixed_7a
        )  # Output channels: 1792

        self.slice4 = nn.Sequential(
            self.inception_resnet.repeat_3,
            self.inception_resnet.block8
        )  # Output channels: 1792

        for param in self.inception_resnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        InceptionResnetV1Outputs = namedtuple(
            "InceptionResnetV1Outputs", ["relu1", "relu2", "relu3", "relu4"]
        )
        out = InceptionResnetV1Outputs(h_relu1, h_relu2, h_relu3, h_relu4)
        return out

def normalize_tensor(x, eps=1e-10): 
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)

def spatial_sum(x, keepdim=True):
    return x.sum([2, 3], keepdim=keepdim)
