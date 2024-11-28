import warnings

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
# from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS


@MODELS.register_module()
class NoFilter(BaseModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

@MODELS.register_module()
class SimpleProjection(BaseModule):
    def __init__(self, in_channels=3, hidden_channels=12, out_channels=3, kernel_size=3, norm_cfg=dict(type='BN')):
        super().__init__()
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            norm_cfg=norm_cfg,
        )
        self.conv2 = ConvModule(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            norm_cfg=norm_cfg,
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


@MODELS.register_module()
class SRMConv2d_simple(BaseModule):

    def __init__(self, inc=3, learnable=False, extra_projection=False):
        super().__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)

        self.extra_projection = extra_projection
        if self.extra_projection:
            self.proj = SimpleProjection(in_channels=inc, hidden_channels=16, out_channels=inc, kernel_size=5)
        # self.hor_kernel = self._build_kernel().transpose(0,1,3,2)

    def forward(self, x):
        '''
        x: imgs (Batch, H, W, 3)
        '''
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        out = self.truc(out)

        if self.extra_projection:
            out = self.proj(out)

        return out

    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # stack the filters
        filters = [[filter1],  # , filter1, filter1],
                   [filter2],  # , filter2, filter2],
                   [filter3]]  # , filter3, filter3]]  # (3,3,5,5)
        filters = np.array(filters)
        filters = np.repeat(filters, inc, axis=1)
        filters = torch.FloatTensor(filters)  # (3,3,5,5)
        return filters






@MODELS.register_module()
class BayarConv2d(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = (torch.ones(self.in_channels, self.out_channels, 1) * -1.000)

        super().__init__()
        # only (kernel_size ** 2 - 1) trainable params as the center element is always -1
        self.kernel = nn.Parameter(torch.rand(self.in_channels, self.out_channels, kernel_size ** 2 - 1),
                                   requires_grad=True)


    def bayarConstraint(self):
        self.kernel.data = self.kernel.permute(2, 0, 1)
        self.kernel.data = torch.div(self.kernel.data, self.kernel.data.sum(0))
        self.kernel.data = self.kernel.permute(1, 2, 0)
        ctr = self.kernel_size ** 2 // 2
        real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]), dim=2)
        real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def forward(self, x):
        if x.shape[1] == 3:
            x = self.rgb2gray(x)
        x = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)

        return x

    def rgb2gray(self, rgb):
        b, g, r = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        gray = torch.unsqueeze(gray, 1)
        return gray


@MODELS.register_module()
class ResidualFilters(BaseModule):
    """
    Residual Filters in preprocessing block
    """

    def __init__(self, inc=3, learnable=False, extra_projection=False):
        super().__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)

        self.extra_projection = extra_projection
        if self.extra_projection:
            self.proj = SimpleProjection(in_channels=inc, hidden_channels=16, out_channels=inc, kernel_size=5)
        # self.hor_kernel = self._build_kernel().transpose(0,1,3,2)

    def forward(self, x):
        '''
        x: imgs (Batch, H, W, 3)
        '''
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        out = self.truc(out)

        if self.extra_projection:
            out = self.proj(out)

        return out

    def _build_kernel(self, inc):

        filter1 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, -1, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter2 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter3 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]

        filter4 = [[0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, -1, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter5 = [[0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, -2, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter6 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]


        filter1 = np.asarray(filter1, dtype=float)
        filter2 = np.asarray(filter2, dtype=float) /2.
        filter3 = np.asarray(filter3, dtype=float) / 4.
        filter4 = np.asarray(filter4, dtype=float)
        filter5 = np.asarray(filter5, dtype=float) / 2.
        filter6 = np.asarray(filter6, dtype=float) / 12.
        # stack the filters
        filters = [[filter1],  # , filter1, filter1],
                   [filter2],  # , filter2, filter2],
                   [filter3],
                   [filter4],
                   [filter5],
                   [filter6]]  # , filter3, filter3]]  # (3,3,5,5)
        filters = np.array(filters)
        filters = np.repeat(filters, inc, axis=1)
        filters = torch.FloatTensor(filters)  # (3,3,5,5)
        return filters



if __name__ ==  '__main__':
    srm = SRMConv2d_simple(inc=3, learnable=False)
