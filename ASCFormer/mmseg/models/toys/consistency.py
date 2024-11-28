import numpy as np
import cv2
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
# from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmseg.registry import MODELS


class ToSimiVolume(nn.Module):
    def __init__(self, in_channels, mode='dot', norm=False, flatten=True):
        '''
        in_channels: number of channels of input
        mode: 'dot' or 'cosine'
        norm: if norm, consistency map ~ [-1,1], otherwise ~ [0,1]
        '''
        super(ToSimiVolume, self).__init__()

        self.in_channels = in_channels
        self.mode = mode
        self.norm = norm
        self.flatten = flatten

        assert self.mode in ['dot', 'cosine']

        self.proj1 = nn.Conv2d(in_channels, in_channels, kernel_size=(1,1))
        self.proj2 = nn.Conv2d(in_channels, in_channels, kernel_size=(1,1))

        # self.sig = nn.Sigmoid()


    def forward(self, x):
        '''
             x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        '''
        B = x.size()[0]

        x_1 = self.proj1(x).view(B, self.in_channels, -1)   # (N, C, T*H*W)
        x_2 = self.proj1(x).view(B, self.in_channels, -1)   # (N, C, T*H*W)
        # x_1 = x_1.permute(0, 2, 1)                          # (N, T*H*W, C)

        if self.mode == 'dot':
            x_1 = x_1.permute(0, 2, 1)                      # (N, T*H*W, C)
            attn = torch.matmul(x_1, x_2)
            attn = attn / math.sqrt(self.in_channels)

            attn = F.sigmoid(attn)
            if self.norm:
                attn = (attn - 0.5) * 2
        else:
            attn = (F.normalize(x_1, dim=1).transpose(-2, -1) @ F.normalize(x_2, dim=1))
            if self.norm:
                pass
            else:
                attn = (attn + 1) / 2

        # contiguous here just allocates contiguous chunk of memory
        y = attn.permute(0, 2, 1).contiguous()

        out = y.view(B, *x.size()[2:], *x.size()[2:])
        if self.flatten:
            out = out.view(B, -1, *x.size()[2:])

        return out