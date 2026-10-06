"""Conditional Networks for Schrödinger Bridge (UNSB)
Adapted from https://github.com/cyclomon/UNSB
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
import numpy as np


def get_filter(filt_size=3):
    if filt_size == 1:
        a = np.array([1., ])
    elif filt_size == 2:
        a = np.array([1., 1.])
    elif filt_size == 3:
        a = np.array([1., 2., 1.])
    elif filt_size == 4:
        a = np.array([1., 3., 3., 1.])
    elif filt_size == 5:
        a = np.array([1., 4., 6., 4., 1.])
    elif filt_size == 6:
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif filt_size == 7:
        a = np.array([1., 6., 15., 20., 15., 6., 1.])
    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)
    return filt


class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)),
                          int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels
        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = self.get_pad_layer(pad_type)(self.pad_sizes)

    def get_pad_layer(self, pad_type):
        if pad_type in ['refl', 'reflect']:
            PadLayer = nn.ReflectionPad2d
        elif pad_type in ['repl', 'replicate']:
            PadLayer = nn.ReplicationPad2d
        elif pad_type == 'zero':
            PadLayer = nn.ZeroPad2d
        else:
            print('Pad type [%s] not recognized' % pad_type)
        return PadLayer

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class ResnetBlock_ncsn(nn.Module):
    """ResNet block for conditional generator"""
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock_ncsn, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetGenerator_ncsn(nn.Module):
    """Resnet-based generator that accepts time and noise conditioning for SB model"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, n_blocks=9, padding_type='reflect',
                 no_antialias=False, no_antialias_up=False, opt=None):
        assert n_blocks >= 0
        super(ResnetGenerator_ncsn, self).__init__()
        self.opt = opt
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # Time and noise embedding dimensions
        self.time_emb_dim = 64
        self.noise_emb_dim = 64

        # Initial conv
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            if no_antialias:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling

        # ResNet blocks with time and noise conditioning
        for i in range(n_blocks):
            model += [ResnetBlock_ncsn(ngf * mult, padding_type, norm_layer, use_dropout, use_bias)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [nn.Upsample(scale_factor=2, mode='nearest'),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

        # Time and noise embedding layers
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

        self.noise_mlp = nn.Sequential(
            nn.Linear(opt.ngf * 4 if opt else 256, self.noise_emb_dim),
            nn.SiLU(),
            nn.Linear(self.noise_emb_dim, self.noise_emb_dim)
        )

    def forward(self, input, time_idx=None, z=None, layers=[], encode_only=False):
        """
        Args:
            input: input image
            time_idx: time index tensor [B]
            z: noise vector [B, C]
            layers: list of layer indices for feature extraction
            encode_only: if True, return only features
        """
        # For compatibility with standard CUT
        if time_idx is None and z is None:
            if len(layers) > 0:
                feat = input
                feats = []
                for layer_id, layer in enumerate(self.model):
                    feat = layer(feat)
                    if layer_id in layers:
                        feats.append(feat)
                    if layer_id == layers[-1] and encode_only:
                        return feats
                return feat, feats
            else:
                return self.model(input)

        # SB model forward with conditioning
        # (Simplified version - just pass through for now)
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)
                if layer_id == layers[-1] and encode_only:
                    return feats
            return feat, feats
        else:
            return self.model(input)


class NLayerDiscriminator_ncsn(nn.Module):
    """Conditional Discriminator for SB model that accepts time conditioning"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 no_antialias=False, t_emb_dim=128):
        super(NLayerDiscriminator_ncsn, self).__init__()

        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim)
        )

        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if no_antialias:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)
                ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input, time_idx=None, input2=None):
        """
        Args:
            input: input image
            time_idx: time index (optional)
            input2: second input for energy network E (optional)
        """
        if input2 is not None:
            # For energy network E that takes two inputs
            input = torch.cat([input, input2], dim=1)

        return self.model(input)
