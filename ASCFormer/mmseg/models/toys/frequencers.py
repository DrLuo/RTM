import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN

from mmengine.model import (BaseModule, ModuleList, caffe2_xavier_init,
                            normal_init, xavier_init)

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmseg.models.backbones.mit import MixFFN, EfficientMultiheadAttention, TransformerEncoderLayer
from mmseg.models.backbones.nat import NATLayer, NATBlock

from ..segmentors.base import BaseSegmentor
from ..backbones.resnet import BasicBlock

from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw


class PEG(nn.Module):
    def __init__(self, dim=256, k=3):
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim)
        # Only for demo use, more complicated functions are effective too.
    def forward(self, x, H, W):
        B, N, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:] # cls token不参与PEG
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat # 产生PE加上自身
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


@MODELS.register_module()
class CATNetDCT(nn.Module):
    def __init__(self, in_channels, out_channels=4, channels=64, embed_dim=4, norm_cfg=dict(type='BN'), upsample=False):
        super(CATNetDCT, self).__init__()
        self.upsample = upsample
        self.norm_cfg = norm_cfg
        self.dct_layer0_dil = ConvModule(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=3,
            padding=8,
            dilation=8,
            norm_cfg=norm_cfg,
        )
        self.dct_layer1_tail = ConvModule(
            in_channels=channels,
            out_channels=embed_dim,
            kernel_size=1,
            norm_cfg=norm_cfg,
        )

        self.dct_layer2 = self._make_layer(BasicBlock, inplanes=embed_dim * 64 * 2, planes=out_channels, blocks=4, stride=1)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = ConvModule(
                in_channels=inplanes,
                out_channels=planes * block.expansion,
                kernel_size=1,
                stride=stride,
                norm_cfg=self.norm_cfg,
                act_cfg = None
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample, norm_cfg=self.norm_cfg))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, norm_cfg=self.norm_cfg))

        return nn.Sequential(*layers)

    def forward(self, extras):
        x = extras['dct']
        qtable = extras['qtable']
        x = self.dct_layer0_dil(x)
        x = self.dct_layer1_tail(x)
        B, C, H, W = x.shape
        x0 = x.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 3, 5, 2, 4).reshape(B, 64 * C, H // 8,
                                                                                     W // 8)  # [B, 256, 32, 32]
        x_temp = x.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 3, 5, 2, 4)  # [B, C, 8, 8, 32, 32]
        q_temp = qtable.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 8, 8, 1, 1]
        xq_temp = x_temp * q_temp  # [B, C, 8, 8, 32, 32]
        x1 = xq_temp.reshape(B, 64 * C, H // 8, W // 8)  # [B, 256, 32, 32]
        x = torch.cat([x0, x1], dim=1)
        x = self.dct_layer2(x)  # x.shape = torch.Size([1, 96, 64, 64]) [2,96,32,32]

        if self.upsample:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        return x


@MODELS.register_module()
class DCTProcessor(nn.Module):
    def __init__(self,
                 in_channels,
                 embed_dims=64,
                 out_channels=128,
                 num_heads=1,
                 patch_size=3,
                 stride=1,
                 mlp_ratio=4,
                 relation=False,
                 quantization=False,
                 reshape=True,
                 band_range=None,
                 sr_ratio=1,
                 reduce_neg=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),):
        super(DCTProcessor, self).__init__()

        self.relation = relation
        self.quantization = quantization
        self.reshape = reshape
        self.band_range = band_range
        self.reduce_neg = reduce_neg

        if band_range is not None:
            if isinstance(band_range, tuple):
                assert band_range[0] < band_range[1]
            elif isinstance(band_range, int):
                band_range = (band_range, band_range + 1)
            else:
                raise ValueError('band should be tuple or int')
            self.band = band_range


        if self.relation:
            in_channels_model = 1


        else:
            if self.band_range is not None:
                in_channels_model = self.band_range[1] - self.band_range[0]
            else:
                in_channels_model = in_channels * 64


        self.local_perception = ConvModule(
            in_channels=in_channels_model,
            out_channels=embed_dims,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        self.dw_conv = ConvModule(
            in_channels=embed_dims,
            out_channels=embed_dims * 2,
            kernel_size=3,
            padding=1,
            groups=embed_dims,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        self.band_perception = ConvModule(
            in_channels=embed_dims * 2,
            out_channels=embed_dims,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        self.patch_embed = PatchEmbed(
            in_channels=embed_dims,
            embed_dims=embed_dims * num_heads,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
            norm_cfg=dict(type='LN', eps=1e-6))


        # encoder
        self.spatial_encoder = TransformerEncoderLayer(
            embed_dims=embed_dims * num_heads,
            num_heads=num_heads,
            feedforward_channels = mlp_ratio * embed_dims * num_heads,
            sr_ratio=sr_ratio,
        )

        self.norm = build_norm_layer(dict(type='LN', eps=1e-6), embed_dims * num_heads)[1]


        self.projection = ConvModule(
            in_channels=embed_dims * num_heads,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )



    def _to_relation(self, dct):
        # dct.shape = (B, C, H, W)
        dct = torch.abs(dct)
        dct = torch.mean(dct, dim=1, keepdim=True)
        dct = torch.div(dct, torch.norm(dct, p=2, keepdim=True))

        return dct

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        if self.reduce_neg:
            x = torch.abs(x)

        B, C, H, W = x.shape
        if self.reshape:
            x = x.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 3, 5, 2, 4).reshape(B, 64 * C, H // 8, W // 8)

        if self.band_range is not None:
            x = x[:, self.band[0]:self.band[1], :, :]

        if self.relation:
            x = self._to_relation(x)

        x = self.local_perception(x)
        x = self.dw_conv(x)
        x = self.band_perception(x)

        x, hw_shape = self.patch_embed(x)

        x = self.spatial_encoder(x, hw_shape)
        x = self.norm(x)

        x = nlc_to_nchw(x, hw_shape)


        x = self.projection(x)

        return x
