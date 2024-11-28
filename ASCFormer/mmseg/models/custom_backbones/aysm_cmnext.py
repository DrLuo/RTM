# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from typing import List, Optional, Tuple, Dict

from mmcv.cnn.bricks import DropPath
import functools
from functools import partial

import torch.utils.checkpoint as cp
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.model.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from mmengine.utils import to_2tuple

from mmseg.models.toys.ffm import FeatureFusionModule as FFM
from mmseg.models.toys.ffm import FeatureRectifyModule as FRM
from mmseg.models.toys.ffm import ChannelEmbed
from mmseg.models.toys.mspa import MSPABlock

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)

from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw
from ..toys.consistency import ToSimiVolume
from ..toys.fusers import NATFuserBlock
from ..backbones.mit import TransformerEncoderLayer
from ..utils import InvertedResidualV3 as InvertedResidual


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)

class MixConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv_3 = nn.Conv2d(
            in_channels // 2,
            out_channels // 2,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=groups // 2,
            bias=bias,
        )
        self.conv_5 = nn.Conv2d(
            in_channels - in_channels // 2,
            out_channels - out_channels // 2,
            kernel_size + 2,
            stride=stride,
            padding=padding + 1,
            dilation=dilation,
            groups=groups - groups // 2,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.conv_3(x1)
        x2 = self.conv_5(x2)
        x = torch.cat([x1, x2], dim=1)
        return x


class MixCFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        feedforward_channels: Optional[int] = None,
        out_features: Optional[int] = None,
        act_func: nn.Module = nn.GELU,
        with_cp: bool = False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = feedforward_channels or in_features
        self.with_cp = with_cp
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.conv = MixConv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
            dilation=1,
            bias=True,
        )
        self.act = act_func()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        def _inner_forward(x: Tensor) -> Tensor:
            x = self.fc1(x)
            B, N, C = x.shape
            x = self.conv(x.transpose(1, 2).view(B, C, H, W))
            x = self.act(x)
            x = self.fc2(x.flatten(2).transpose(-1, -2))
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x, H, W):
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))


class MixFFN(BaseModule):
    """An implementation of MixFFN of Segformer.

    The differences between MixFFN & FFN:
        1. Use 1X1 Conv to replace Linear layer.
        2. Introduce 3X3 Conv to encode positional information.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class ChannelProcessing(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., drop_path=0., mlp_hidden_dim=None,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_v = MLP(dim, mlp_hidden_dim)
        self.norm_v = norm_layer(dim)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.pool = nn.AdaptiveAvgPool2d((None, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, H, W, atten=None):
        B, N, C = x.shape

        v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q.softmax(-2).transpose(-1, -2)
        _, _, Nk, Ck = k.shape
        k = k.softmax(-2)
        k = torch.nn.functional.avg_pool2d(k, (1, Ck))

        attn = self.sigmoid(q @ k)

        Bv, Hd, Nv, Cv = v.shape
        v = self.norm_v(self.mlp_v(v.transpose(1, 2).reshape(Bv, Nv, Hd * Cv), H, W)).reshape(Bv, Nv, Hd, Cv).transpose(
            1, 2)
        x = (attn * v.transpose(-1, -2)).permute(0, 3, 1, 2).reshape(B, N, C)
        return x


class PredictorConv(nn.Module):
    def __init__(self, embed_dim=384, num_modals=4):
        super().__init__()
        self.num_modals = num_modals
        self.score_nets = nn.ModuleList([nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=(embed_dim)),
            nn.Conv2d(embed_dim, 1, 1),
            nn.Sigmoid()
        ) for _ in range(num_modals)])

    def forward(self, x):
        B, C, H, W = x[0].shape
        x_ = [torch.zeros((B, 1, H, W)) for _ in range(self.num_modals)]
        for i in range(self.num_modals):
            x_[i] = self.score_nets[i](x[i])
        return x_


# TODO: Make it theoretically correct
class ModalSelector(nn.Module):
    def __init__(self, embed_dims=128, num_modals=4, sr_ratio=1):
        super().__init__()
        self.num_modals = num_modals
        self.simi_projs = nn.ModuleList([
            nn.ModuleList([
                ToSimiVolume(embed_dims, mode='dot', norm=True),
                nn.Sequential(
                    nn.Conv2d(4, 4, 1),
                    nn.ReLU(),
                )
            ]) for _ in range(num_modals)])

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr_block = nn.ModuleList([
                nn.ModuleList([
                    Conv2d(
                        in_channels=embed_dims,
                        out_channels=embed_dims,
                        kernel_size=sr_ratio,
                        stride=sr_ratio),
                    nn.LayerNorm(embed_dims),
                ]) for _ in range(num_modals)])

        self.channel_mixer = nn.Sequential(
            nn.Conv2d(4 * num_modals, 4 * num_modals * 4, 1),
            nn.ReLU(),
            nn.Conv2d(4 * num_modals * 4, num_modals, 1),
        )


    def forward(self, x):
        B, C, H, W = x[0].shape
        rs_shape = (H // self.sr_ratio, W // self.sr_ratio)

        simi_vols = []
        for i in range(self.num_modals):
            temp = x[i]
            if self.sr_ratio > 1:
                temp = self.sr_block[i][0](temp)
                temp = nchw_to_nlc(temp)
                temp = self.sr_block[i][1](temp)
                temp = nlc_to_nchw(temp, rs_shape)
            temp = self.simi_projs[i][0](temp)
            simi_vols.append([])
            simi_vols[i].append(torch.mean(temp, dim=1, keepdim=True))
            simi_vols[i].append(torch.var(temp, dim=1, keepdim=True))
            simi_vols[i].append(torch.max(temp, dim=1, keepdim=True)[0])
            simi_vols[i].append(torch.min(temp, dim=1, keepdim=True)[0])

            simi_vols[i] = self.simi_projs[i][1](torch.cat(simi_vols[i], dim=1))

        out = self.channel_mixer(torch.cat(simi_vols, dim=1))
        out = torch.softmax(out, dim=1)

        if self.sr_ratio > 1:
            out = F.interpolate(out, size=(H, W), mode='nearest')

        return out


class ModalSelectorV2(nn.Module):
    def __init__(self, embed_dims=128, num_modals=4, sr_ratio=1):
        super().__init__()
        self.num_modals = num_modals
        self.simi_projs = nn.ModuleList([
            nn.ModuleList([
                ToSimiVolume(embed_dims, mode='cosine', norm=True),
                # nn.Sequential(
                #     nn.Conv2d(4, 4, 1),
                #     nn.ReLU(),
                # )
            ]) for _ in range(num_modals)])

        self.nonlinear = nn.Sequential(
            nn.Conv2d(4, 4 * 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(4 * 4, 1, 1),
        )

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr_block = nn.ModuleList([
                nn.ModuleList([
                    Conv2d(
                        in_channels=embed_dims,
                        out_channels=embed_dims,
                        kernel_size=sr_ratio,
                        stride=sr_ratio),
                    nn.LayerNorm(embed_dims),
                ]) for _ in range(num_modals)])


    def forward(self, x):
        B, C, H, W = x[0].shape
        rs_shape = (H // self.sr_ratio, W // self.sr_ratio)

        simi_vols = []
        for i in range(self.num_modals):
            temp = x[i]
            if self.sr_ratio > 1:
                temp = self.sr_block[i][0](temp)
                temp = nchw_to_nlc(temp)
                temp = self.sr_block[i][1](temp)
                temp = nlc_to_nchw(temp, rs_shape)
            temp = self.simi_projs[i][0](temp)
            simi_vols.append([])
            simi_vols[i].append(torch.mean(temp, dim=1, keepdim=True))
            simi_vols[i].append(torch.var(temp, dim=1, keepdim=True))
            simi_vols[i].append(torch.max(temp, dim=1, keepdim=True)[0])
            simi_vols[i].append(torch.min(temp, dim=1, keepdim=True)[0])

            simi_vols[i] = self.nonlinear(torch.cat(simi_vols[i], dim=1))

        out = torch.cat(simi_vols, dim=1)
        out = torch.softmax(out, dim=1)

        if self.sr_ratio > 1:
            out = F.interpolate(out, size=(H, W), mode='nearest')

        return out


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class ConvLayerNorm(nn.Module):
    """Channel first layer norm
    """

    def __init__(self, normalized_shape, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class LayerNormParallel(nn.Module):
    def __init__(self, num_features, num_modals=4):
        super(LayerNormParallel, self).__init__()
        # self.num_modals = num_modals
        for i in range(num_modals):
            setattr(self, 'ln_' + str(i), ConvLayerNorm(num_features, eps=1e-6))

    def forward(self, x_parallel):
        return [getattr(self, 'ln_' + str(i))(x) for i, x in enumerate(x_parallel)]


class PatchEmbedParallel(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4, padding=0, num_modals=4):
        super().__init__()
        self.proj = ModuleParallel(nn.Conv2d(c1, c2, patch_size, stride, padding))  # padding=(ps[0]//2, ps[1]//2)
        self.norm = LayerNormParallel(c2, num_modals)

    def forward(self, x: list) -> list:
        x = self.proj(x)
        _, _, H, W = x[0].shape
        x = self.norm(x)
        return x, H, W




class DetailedPatchEmbedParallel(nn.Module):
    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 kernel_size=7,
                 stride=None,
                 dilation=1,
                 num_modals=4,
                 to_hw=True,
                 norm_cfg=dict(type='LN')):
        super(DetailedPatchEmbedParallel, self).__init__()

        assert num_modals > 0
        self.to_hw = to_hw
        self.projs = []
        for i in range(num_modals):
            self.projs.append(
                PatchEmbed(
                    in_channels=in_channels,
                    embed_dims=embed_dims,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    dilation=dilation,
                    norm_cfg=norm_cfg)
            )
        self.projs = nn.ModuleList(self.projs)


    def forward(self, x: list):
        outs = []
        for i in range(len(x)):
            out, hw_shape = self.projs[i](x[i])
            if self.to_hw:
                outs.append(nlc_to_nchw(out, hw_shape))
            else:
                outs.append(out)

        return outs, hw_shape






# TODO: Neighborhood Attention Based Rectifier
class NABR(nn.Module):
    def __init__(self, embed_dim=128, num_modals=4):
        super().__init__()
        pass
    def forward(self, x):
        pass


class InvertedResidualParallel(nn.Module):
    def __init__(self, num_modals=4, **kwargs):
        super().__init__()
        self.num_modals = num_modals
        self.blocks = nn.ModuleList([InvertedResidual(**kwargs) for _ in range(num_modals)])

    def forward(self, x_parallel):
        out = []
        for i in range(len(x_parallel)):
            out.append(self.blocks[i](x_parallel[i]))

        return out


class InvertedResidualSiamese(nn.Module):
    def __init__(self, num_modals=4, **kwargs):
        super().__init__()
        self.num_modals = num_modals
        self.blocks = ModuleParallel(InvertedResidual(**kwargs))

    def forward(self, x_parallel):
        return self.blocks(x_parallel)




@MODELS.register_module()
class PPXVisionTransformer(BaseModule):
    """The backbone of Segformer.

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 modals=['dct', 'srm'],
                 in_modals=None,
                 skip_patch_embed_stage=-1,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratios=(8,8,4,4),
                 # qkv_bias=True,
                 drop_rate=0.,
                 # attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 # act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.in_channels = in_channels
        self.modals = modals
        self.num_modals = len(modals)

        self.in_modals = in_modals
        self.skip_patch_embed_stage = skip_patch_embed_stage

        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        if self.in_modals is not None:
            assert max(self.in_modals) <= self.num_modals
            assert len(self.in_modals) == self.num_stages
        else:
            self.in_modals = [self.num_modals] * self.num_stages

        cur = 0
        self.layers = ModuleList()
        self.extra_score_predictor = ModuleList([])
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = PatchEmbedParallel(
                c1=in_channels,
                c2=embed_dims_i,
                patch_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2,
                num_modals=self.in_modals[i-1] if (i == self.skip_patch_embed_stage) else self.in_modals[i],
            )
            if self.in_modals[i] > 1:
                self.extra_score_predictor.append(PredictorConv(embed_dims_i, self.in_modals[i]))
            layer = ModuleList([
                MSPABlock(
                    dim=embed_dims_i,
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + idx]) for idx in range(num_layer)
            ])
            in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = ConvLayerNorm(embed_dims_i)
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super().init_weights()

    def forward(self, x):
        outs = []


        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            if self.in_modals[i] > 1:
                x = self.tokenselect(x, self.extra_score_predictor[i]) if self.in_modals[i] > 1 else x[0]
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)

        return outs

    def tokenselect(self, x_ext, module):
        x_scores = module(x_ext)                            #
        for i in range(len(x_ext)):
            x_ext[i] = x_scores[i] * x_ext[i] + x_ext[i]    # 加权
        x_f = functools.reduce(torch.max, x_ext)
        return x_f



@MODELS.register_module()
class HubVisionTransformer(BaseModule):
    """The backbone of Segformer.

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 modals=['dct', 'srm'],
                 in_modals=None,
                 skip_patch_embed_stage=-1,
                 modal_interact=False,
                 modals_proj = False,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.in_channels = in_channels
        self.modals = modals
        self.num_modals = len(modals)

        self.in_modals = in_modals
        self.skip_patch_embed_stage = skip_patch_embed_stage
        self.modal_interact = modal_interact
        self.modals_proj = modals_proj

        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        # if self.num_modals > 1:
        #     self.extra_score_predictor = nn.ModuleList([PredictorConv(embed_dims * num_heads[i], self.num_modals) for i in range(len(num_layers))])
        if self.in_modals is not None:
            assert max(self.in_modals) <= self.num_modals
            assert len(self.in_modals) == self.num_stages
        else:
            self.in_modals = [self.num_modals] * self.num_stages

        cur = 0
        self.layers = ModuleList()
        self.extra_score_predictor = ModuleList([])
        # self.modal_channel_mixers = ModuleList([])
        self.modal_convs = ModuleList([])
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = DetailedPatchEmbedParallel(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                # padding=patch_sizes[i] // 2,
                num_modals=self.in_modals[i-1] if (i == self.skip_patch_embed_stage) else self.in_modals[i],
            )
            if self.in_modals[i] > 1:
                self.extra_score_predictor.append(ModalSelectorV2(embed_dims_i, self.in_modals[i], sr_ratios[i]))
                # self.modal_channel_mixers.append(nn.ModuleList([nn.Conv2d(embed_dims_i, embed_dims_i, kernel_size=1, stride=1, padding=0) for _ in range(self.in_modals[i])]))
            else:
                self.extra_score_predictor.append(nn.Identity())

            if self.modals_proj:
                self.modal_convs.append(
                    InvertedResidualParallel(
                        in_channels=embed_dims_i,
                        out_channels=embed_dims_i,
                        mid_channels= embed_dims_i * 2,
                        se_cfg = dict(
                            channels=embed_dims_i * 2,
                            ratio=4,
                            act_cfg=(dict(type='ReLU'),
                                     dict(type='HSigmoid', bias=3.0, divisor=6.0)))
                    )
                )
            else:
                self.modal_convs.append((AnyIdentity()))


            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super().init_weights()

    def forward(self, x):
        outs = []

        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            if self.in_modals[i] > 1:
                x = self.tokenselect(x, self.extra_score_predictor[i]) if self.in_modals[i] > 1 else x[0]
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)

        return outs


    def tokenselect(self, x_ext, module):
        if len(x_ext) == 1:
            x_f = x_ext[0]

        else:
            x_scores = module(x_ext)    # [B, num_modals, H, W]
            x_scores = x_scores.unsqueeze(2)    # [B, num_modals, 1, H, W]
            x_scores = x_scores.transpose(0, 1)    # [num_modals, B, 1, H, W]

            # print(x_scores.shape, x_ext[0].shape)

            for i in range(len(x_ext)):
                x_ext[i] = x_scores[i] * x_ext[i] + x_ext[i]    # 加权
            # x_f = functools.reduce(torch.max, x_ext)
            x_f = torch.sum(torch.stack(x_ext), dim=0)    # [B, C, H, W]

        x_f = x_f.flatten(2).transpose(1, 2)

        return x_f


class AnyIdentity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *kwargs):
        return kwargs


@MODELS.register_module()
class AsymCMNeXtV2(BaseModule):
    """The backbone of CMNeXt but allow asymmetric input.

    This backbone is the Upgrade of `CMNeXt:'
    Delivering Arbitrary-Modal Semantic Segmentation
    modified from SegFormer

    """
    def __init__(self,
                 backbone_main: ConfigType,
                 backbone_extra: ConfigType,
                 use_rectifier: bool,
                 rectifier: OptConfigType=None,
                 fuser: OptConfigType=None,
                 num_heads=[1,2,5,8],
                 in_stages=None,
                 extra_patch_embed: dict=None,
                 out_indices=(0, 1, 2, 3),
                 init_cfg=None,
                 spatial_reshape=False,
                 no_select=False,
                 ):
        super().__init__(init_cfg=init_cfg)

        self.out_indices = out_indices
        self.spatial_reshape = spatial_reshape
        self.use_rectifier = use_rectifier
        self.fuser = fuser
        self.no_select = no_select

        self.main_branch = MODELS.build(backbone_main)
        self.extra_branch = MODELS.build(backbone_extra)

        if in_stages is not None:
            assert len(in_stages) == self.extra_branch.num_modals
        self.in_stages = in_stages
        if extra_patch_embed is not None:
            self.extra_patch_embed = PatchEmbed(
                in_channels=extra_patch_embed['in_channels'],
                embed_dims=extra_patch_embed['embed_dims'],
                kernel_size=extra_patch_embed['kernel_size'],
                stride=extra_patch_embed['stride'],
                padding=extra_patch_embed['kernel_size'] // 2,
                norm_cfg=dict(type='LN', eps=1e-6),
            )
            self.use_extra_patch_embed = True
            self.reshape_extra_nchw = extra_patch_embed['reshape']
        else:
            self.use_extra_patch_embed = False


        self.num_stage_main = self.main_branch.num_layers.__len__()
        self.num_stage_extra = self.extra_branch.num_layers.__len__()

        assert self.num_stage_main >= self.num_stage_extra, 'main branch must have more stages than extra branch'
        self.shift_stage = self.num_stage_main - self.num_stage_extra


        num_heads = self.extra_branch.num_heads

        # fusion module
        self.FRMs = []


        # feature rectification module
        self.FFMs = []

        # pre-fusion module
        embed_dims = [self.extra_branch.embed_dims * num_heads[i] for i in range(len(num_heads))]

        for i in range(len(num_heads)):
            if self.use_rectifier:
                self.FRMs.append(FRM(dim=embed_dims[i], reduction=1))
            else:
                self.FRMs.append(AnyIdentity())
            if self.fuser is None:
                self.FFMs.append(
                    NATFuserBlock(
                        a_channel=embed_dims[i],
                        b_channel=embed_dims[i],
                        num_head=self.extra_branch.num_heads[i],
                        kernel_size=5,
                        gated=True,
                    )
                )
            elif isinstance(self.fuser, dict):
                self.fuser.update(dict(a_channel=embed_dims[i], b_channel=embed_dims[i]))
                if self.fuser['type'] == 'NATFuserBlock':
                    self.fuser.update(dict(num_head=self.extra_branch.num_heads[i]))
                elif self.fuser['type'] == 'EfficientAttentionFuserBlock':
                    self.fuser.update(dict(num_head=self.extra_branch.num_heads[i], sr_ratio=self.extra_branch.sr_ratios[i]))
                print(self.fuser)
                fuser_block = MODELS.build(self.fuser)
                self.FFMs.append(
                    fuser_block
                )


        self.FRMs = nn.ModuleList(self.FRMs)
        self.FFMs = nn.ModuleList(self.FFMs)


    def merge_inputs(self, x_a, x_b):
        if isinstance(x_a, list):
            merged = x_a
        else:
            merged = [x_a]

        if isinstance(x_b, list):
            merged = merged.extend(x_b)
        else:
            merged = merged.append(x_b)

        return merged

    def forward(self, x):
        outs = []

        x_cam = x[0]
        x_extra = x[1]

        # align channel
        for i in range(len(x_extra)):
            if x_extra[i].size()[1] != self.extra_branch.in_channels:
                if (x_extra[i].size()[1] == 1) & (x_cam.size()[2:] == x_extra[i].size()[2:]):
                    x_extra[i] = x_extra[i].repeat(1, self.extra_branch.in_channels, 1, 1)

        if self.in_stages is not None:
            if max(self.in_stages) > 0:
                in_stage = max(self.in_stages)

                x_extra_0 = []
                x_extra_m = []
                for i in range(len(self.in_stages)):
                    if self.in_stages[i] > 0:
                        x_extra_m.append(x_extra[i])
                    else:
                        x_extra_0.append(x_extra[i])

                x_extra = x_extra_0
            else:
                in_stage = -1
        else:
            in_stage = -1

        B = x_cam.shape[0]

        for i in range(self.shift_stage):
            layer = self.main_branch.layers[i]
            x_cam, hw_shape = layer[0](x_cam)
            for block in layer[1]:
                x_cam = block(x_cam, hw_shape)
            x_cam = layer[2](x_cam)
            x_cam = nlc_to_nchw(x_cam, hw_shape)
            if i in self.out_indices:
                outs.append(x_cam)

        # cross recalibration
        for i in range(self.num_stage_extra):
            layer = self.main_branch.layers[(i+self.shift_stage)]
            layer_extra = self.extra_branch.layers[i]
            x_cam, hw_shape = layer[0](x_cam)
            H, W = hw_shape
            for block in layer[1]:
                x_cam = block(x_cam, hw_shape)
            x_cam = layer[2](x_cam)
            x_cam = nlc_to_nchw(x_cam, hw_shape)

            if self.in_stages is not None:
                if i == in_stage:
                    if self.use_extra_patch_embed:
                        if self.reshape_extra_nchw:
                            temp = []
                            for item in x_extra_m:
                                em, s = self.extra_patch_embed(item)
                                temp.append(nlc_to_nchw(em, s))
                            x_extra_m = temp
                        else:
                            x_extra_m = [self.extra_patch_embed(item)[0] for item in x_extra_m]

                        x_extra, _ = layer_extra[0](x_extra)
                        x_extra.extend(x_extra_m)
                    else:
                        x_extra.extend(x_extra_m)
                        x_extra, _ = layer_extra[0](x_extra)
                else:
                    x_extra, _ = layer_extra[0](x_extra)

            else:
                x_extra, _ = layer_extra[0](x_extra)

            if self.no_select:
                x_f = torch.stack(x_extra,dim=0).mean(dim=0)
                x_f = x_f.flatten(2).transpose(1, 2)
            else:
                x_f = self.extra_branch.tokenselect(x_extra, self.extra_branch.extra_score_predictor[i])

            for block in layer_extra[1]:
                x_f = block(x_f, hw_shape)
            x_f = layer_extra[2](x_f)
            x_f = nlc_to_nchw(x_f, hw_shape)

            x, x_f = self.FRMs[i](x_cam, x_f)
            x_fused = self.FFMs[i](x_cam, x_f)

            if (i+self.shift_stage) in self.out_indices:
                outs.append(x_fused)

            x_extra = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2) + x_f for x_ in x_extra] if self.extra_branch.num_modals > 1 else [x_f]

        return outs


