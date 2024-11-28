# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize

from pathlib import Path
import numpy as np
import pickle as pkl
import os


@MODELS.register_module()
class SegformerHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', save_feat=False, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        self.save_feat = save_feat
        if self.save_feat:
            self.save_dir = 'vis/feat'
        else:
            self.save_dir = None

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.cls_seg(out)

        return out

    def forward_infer(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []

        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

            # feats.append(
            #     self.projs[idx][1](self.projs[idx][0](outs[idx])))

        feats = self.fusion_conv(torch.cat(outs, dim=1))
        # feats = out.detach()

        # mid = self.seg_proj(out)
        out = self.cls_seg(feats)

        return out, feats

    def predict(self, inputs, batch_img_metas, test_cfg):
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        if self.save_feat:
            seg_logits, feats = self.forward_infer(inputs)
            self.save_deep_feature(feats.detach(), self.save_dir, batch_img_metas)
        else:
            seg_logits = self.forward(inputs)

        return self.predict_by_feat(seg_logits, batch_img_metas)

    def save_deep_feature(self, feat, save_dir, batch_img_metas):
        """save intermediate feature map"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # for i in range(len(batch_img_metas)):
        feat = feat[0].detach().permute(1,2,0).cpu().numpy()
        img_meta = batch_img_metas[0]
        img_name = img_meta['img_path']
        img_name = os.path.basename(img_name).split('.')[0]
        # img_name = img_name.split('/')[-1].split('.')[0]
        img_name = img_name + '.npy'
        np.save(os.path.join(save_dir, img_name), feat)
        print('save feature map: ', img_name)
