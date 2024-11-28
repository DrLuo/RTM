import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


from mmcv.cnn import ConvModule
from mmcv.cnn import Conv2d, build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.cnn.bricks.transformer import FFN, build_dropout

from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.model.weight_init import xavier_init
from mmengine.utils import to_2tuple

from mmseg.registry import MODELS
from ..utils import resize
from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw


from torch.nn.functional import pad
from torch.nn.init import trunc_normal_

from ..backbones.nat import NATLayer, ConvTokenizer

from natten.functional import natten2dav, natten2dqkrpb
from natten import NeighborhoodAttention2D as NeighborhoodAttention
from timm.models.layers import DropPath

class NATFusionModule(nn.Module):
    """
    Feature Fusion based on Neighborhood attention.
    Modified from Neighborhood Attention 2D Module
    """

    def __init__(
        self,
        dim,
        num_heads,
        kernel_size,
        dilation=1,
        bias=True,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_mode='cross',
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        assert attn_mode in ['self', 'cross', 'mix', 'recross']
        self.attn_mode = attn_mode

        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
            dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.embed_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.embed_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.embed_v = nn.Linear(dim, dim, bias=qkv_bias)

        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        # x: [B, H, W, C] is the main feature map
        # y: [B, H, W, C] is the auxiliary feature map

        B, Hp, Wp, C = x.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.window_size or W < self.window_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.window_size - W)
            pad_b = max(0, self.window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape

        if self.attn_mode == 'cross':
            q = self.embed_q(x).reshape(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
            k = self.embed_k(y).reshape(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
            v = self.embed_v(y).reshape(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        elif self.attn_mode == 'self':
            q = self.embed_q(y).reshape(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
            k = self.embed_k(y).reshape(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
            v = self.embed_v(x).reshape(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        elif self.attn_mode == 'recross':
            q = self.embed_q(y).reshape(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
            k = self.embed_k(x).reshape(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
            v = self.embed_v(x).reshape(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        else:
            raise NotImplementedError

        q = q * self.scale
        attn = natten2dqkrpb(q, k, self.rpb, self.kernel_size, self.dilation)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = natten2dav(attn, v, self.kernel_size, self.dilation)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"rel_pos_bias={self.rpb is not None}"
        )


class NATFusionLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=7,
        dilation=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
        attn_mode='cross',
        gated=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.gated = gated

        self.norm1_a = norm_layer(dim)
        self.norm1_b = norm_layer(dim)
        self.attn = NATFusionModule(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_mode=attn_mode,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )

        if self.gated:
            self.gate_para = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1_a(x)
            y = self.norm1_b(y)
            a = self.attn(x, y)
            if self.gated:
                x = (1-F.tanh(self.gate_para)) * shortcut + F.tanh(self.gate_para) * self.drop_path(a)
            else:
                x = shortcut + self.drop_path(a)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1_a(x)
        y = self.norm1_b(y)
        a = self.attn(x, y)
        if self.gated:
            x = (1-F.tanh(self.gate_para)) * shortcut + F.tanh(self.gate_para) * self.drop_path(self.gamma1 * a)
        else:
            x = shortcut + self.drop_path(self.gamma1 * a)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x



@MODELS.register_module()
class NATFuserBlock(nn.Module):
    """
    NATFuserBlock for NATFuser, self -> cross
    """
    def __init__(
            self,
            a_channel,
            b_channel,
            num_head,
            kernel_size,
            dilation=None,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            layer_scale=None,
            attn_mode='cross',
            num_layers=1,
            expand_rate=1,
            gated=False,
            post_attn=False,
    ):
        super().__init__()
        assert isinstance(a_channel, int)
        assert isinstance(b_channel, int)

        self.a_channel = a_channel
        self.b_channel = b_channel
        self.attn_mode = attn_mode
        self.gated = gated
        self.post_attn = post_attn
        self.num_layers = num_layers

        # self + cross
        # Neighborhood self-attention
        self.local_self_attn_a = NATLayer(
                dim=a_channel,
                num_heads=num_head,
                kernel_size=kernel_size,
                dilation=None if dilation is None else dilation,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                norm_layer=norm_layer,
                layer_scale=layer_scale,
            )


        self.local_self_attn_b = NATLayer(
                dim=b_channel,
                num_heads=num_head,
                kernel_size=kernel_size,
                dilation=None if dilation is None else dilation,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                norm_layer=norm_layer,
                layer_scale=layer_scale,
            )

        if self.post_attn:
            # Neighborhood self-attention
            self.post_self_attn = NATLayer(
                dim=a_channel,
                num_heads=num_head,
                kernel_size=kernel_size,
                dilation=None if dilation is None else dilation,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                norm_layer=norm_layer,
                layer_scale=layer_scale,
            )

        # Neighborhood cross-attention
        self.local_cross_attn = nn.ModuleList([
            NATFusionLayer(
                dim=b_channel,
                num_heads=num_head,
                kernel_size=kernel_size,
                dilation=None if dilation is None else dilation,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                norm_layer=norm_layer,
                layer_scale=layer_scale,
                gated=gated,
            ) for _ in range(num_layers)
        ])


        self.norm_layers = nn.ModuleList()

        layer = norm_layer(a_channel)
        layer_name = f"norm"
        self.add_module(layer_name, layer)

    def forward(self, x, y):

        a = x.permute(0, 2, 3, 1)
        a = self.local_self_attn_a(a)

        b = y.permute(0, 2, 3, 1)
        b = self.local_self_attn_b(b)


        for i in range(self.num_layers):
            a = self.local_cross_attn[i](a, b)

        if self.post_attn:
            a = self.post_self_attn(a)

        norm_layer = getattr(self, f"norm")
        out = norm_layer(a).permute(0, 3, 1, 2).contiguous()

        return out





@MODELS.register_module()
class NATFuser(nn.Module):
    """
    Feature Fusion based on Neighborhood attention.
    Modified from Neighborhood Attention 2D Module
    """

    def __init__(
        self,
        a_channels,
        b_channels,
        num_heads,
        kernel_size,
        dilations=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
        attn_mode='cross',
        num_layers=1,
        gated=False,
    ):
        super().__init__()
        assert isinstance(a_channels, list)
        assert isinstance(b_channels, list)
        assert len(a_channels) >= len(b_channels)

        self.a_channels = a_channels
        self.b_channels = b_channels
        self.attn_mode = attn_mode
        self.num_layers = num_layers

        self.gated = gated

        self.shift = len(a_channels) - len(b_channels)

        # self + cross
        self.local_self_attn_a = nn.ModuleList()
        self.local_self_attn_b = nn.ModuleList()
        self.local_cross_attn = nn.ModuleList()


        # Neighborhood self-attention
        for i in range(len(a_channels)):
            self.local_self_attn_a.append(
                NATLayer(
                    dim=a_channels[i],
                    num_heads=num_heads[i],
                    kernel_size=kernel_size,
                    dilation=None if dilations is None else dilations[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    layer_scale=layer_scale,
                )
            )

        for i in range(len(b_channels)):
            self.local_self_attn_b.append(
                NATLayer(
                    dim=b_channels[i],
                    num_heads=num_heads[self.shift + i],
                    kernel_size=kernel_size,
                    dilation=None if dilations is None else dilations[self.shift + i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[self.shift + i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    layer_scale=layer_scale,
                )
            )

        # Neighborhood cross-attention
        for i in range(len(b_channels)):
            if num_layers > 1:
                self.local_cross_attn.append(
                    nn.ModuleList([
                        NATFusionLayer(
                            dim=b_channels[i],
                            num_heads=num_heads[self.shift + i],
                            kernel_size=kernel_size,
                            dilation=None if dilations is None else dilations[self.shift + i],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop=drop,
                            attn_drop=attn_drop,
                            drop_path=drop_path[self.shift + i]
                            if isinstance(drop_path, list)
                            else drop_path,
                            norm_layer=norm_layer,
                            layer_scale=layer_scale,
                            gated=gated,
                        ) for _ in range(num_layers)
                    ])
                )
            else:
                self.local_cross_attn.append(
                    NATFusionLayer(
                        dim=b_channels[i],
                        num_heads=num_heads[self.shift + i],
                        kernel_size=kernel_size,
                        dilation=None if dilations is None else dilations[self.shift + i],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[self.shift + i]
                        if isinstance(drop_path, list)
                        else drop_path,
                        norm_layer=norm_layer,
                        layer_scale=layer_scale,
                        gated=gated,
                    )
                )

        self.norm_layers = nn.ModuleList()

        for i in range(len(a_channels)):
            layer = norm_layer(a_channels[i])
            layer_name = f"norm{i}"
            self.add_module(layer_name, layer)


    def forward(self, x, y):
        num_x = len(x)
        num_y = len(y)

        shift = num_x - num_y
        assert shift >= 0

        outs = []

        for i in range(num_x):
            a = x[i].permute(0, 2, 3, 1)
            a = self.local_self_attn_a[i](a)
            # if i < shift:
            outs.append(a)

        for i in range(num_y):
            a = outs[i + shift]
            b = y[i].permute(0, 2, 3, 1)

            b = self.local_self_attn_b[i](b)
            if self.num_layers > 1:
                for j in range(self.num_layers):
                    a = self.local_cross_attn[i][j](a, b)
            else:
                a = self.local_cross_attn[i](a, b)
            outs[i + shift] = a

        for i in range(len(outs)):
            norm_layer = getattr(self, f"norm{i}")
            outs[i] = norm_layer(outs[i]).permute(0, 3, 1, 2).contiguous()

        return outs


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



