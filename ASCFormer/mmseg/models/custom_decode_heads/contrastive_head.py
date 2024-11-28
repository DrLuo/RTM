# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import build_conv_layer, build_norm_layer
from lightly import loss as loss_contrastive
# from lightly.models.modules import heads

from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize
from ..builder import build_loss

from pathlib import Path
import numpy as np
import pickle as pkl
import os


@MODELS.register_module()
class ContrastiveHeadV2(BaseDecodeHead):
    """Modified from SegFormer Head.

    My decode head for tampered text detection in images.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self,
                 interpolate_mode='bilinear',
                 use_cl=True,
                 cl_sampler='random',
                 batch_cl=True,
                 field_mode=None,
                 dim = 128,
                 max_points=1024,
                 min_points=2,
                 multi_layer_cl=False,
                 save_feat=False,
                 upsample_first=False,
                 use_memory=True,
                 max_memory_step=3,
                 max_memory_size=None,
                 up_decode=False,
                 loss_const = dict(
                     type='SupDCLLoss',
                     loss_weight=1.0),
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        self.use_cl = use_cl
        self.cl_sampler = cl_sampler
        if cl_sampler is not None:
            assert cl_sampler in ['ori', 'all', 'random', 'balance', 'edge', 'weighted', 'hard', 'hard_a', 'hard_b']

        self.max_points = max_points
        self.multi_layer_cl = multi_layer_cl
        self.up_sample_first = upsample_first

        self.all_points = 2 * max_points
        # self.min_points = min_points
        self.batch_cl = batch_cl
        self.up_decode = up_decode
        self.save_feat = save_feat
        if self.save_feat:
            self.save_dir = 'vis/feat'
        else:
            self.save_dir = None

        self.use_memory = use_memory
        if use_memory:
            self.memory_bank = MemoryBank(max_memory_step, 512 if max_memory_size is None else max_memory_size)

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
            kernel_size=3,
            padding=1,
            norm_cfg=self.norm_cfg)

        if self.use_cl:
            self.cc_proj = nn.Sequential(
                ConvModule(
                    in_channels=self.channels,
                    out_channels=dim*4,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg),
                ConvModule(
                    in_channels=dim*4,
                    out_channels=dim,
                    kernel_size=1,
                    norm_cfg=None,
                    act_cfg=None)
                )

            if isinstance(loss_const, dict):
                self.loss_const = build_loss(loss_const)
            else:
                raise NotImplementedError

        self.seg_proj = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            norm_cfg=self.norm_cfg)



    def init_memory_bank(self):
        if getattr(self, 'memory_bank', None) is not None:
            self.memory_bank.reset_bank()

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

        if self.use_cl:
            feats = out
            if self.up_sample_first:
                feats = resize(
                    input=feats,
                    size=[s * 2 for s in feats.shape[2:]],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners)
            feats = self.cc_proj(feats)
        else:
            feats = None

        if self.up_decode:
            out = resize(
                input=out,
                size=[s * 2 for s in out.shape[2:]],
                mode=self.interpolate_mode,
                align_corners=self.align_corners)
        out = self.seg_proj(out)
        out = self.cls_seg(out)

        return out, feats

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

        out = self.fusion_conv(torch.cat(outs, dim=1))
        feats = out.detach()

        mid = self.seg_proj(out)
        out = self.cls_seg(mid)

        return out, feats


    def loss(self, inputs, batch_data_samples,
             train_cfg) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits, feats = self.forward(inputs)

        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        if self.use_cl:
            loss_cl = self.loss_by_cl(feats, batch_data_samples)
            if loss_cl['loss_contrastive'] is not None:
                losses.update(loss_cl)


        return losses

    def loss_by_cl(self, feats, batch_data_samples):

        B, C, H, W = feats.size()
        seg_label = self._stack_batch_gt(batch_data_samples)

        if self.up_sample_first:
            feats = resize(
                input=feats,
                size=seg_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        seg_label = F.interpolate(seg_label.float(), size=(H, W), mode='nearest')

        pos_feats, neg_feats = self.sample_pairs(feats, seg_label, self.cl_sampler, batch_cl=self.batch_cl)

        if self.batch_cl:
            sampled_feats = torch.cat([pos_feats, neg_feats], dim=0)
            sampled_labels = torch.cat([torch.ones(pos_feats.size(0)), torch.zeros(neg_feats.size(0))],
                                       dim=0).long().to(feats.device).unsqueeze(1)

            loss_cl = self.loss_const(sampled_feats, sampled_labels)

        else:
            valid_num = len(pos_feats)
            loss_cl = torch.tensor(0.).to(feats.device)

            for pos, neg in zip(pos_feats, neg_feats):
                sampled_feats = torch.cat([pos, neg], dim=0)
                sampled_labels = torch.cat([torch.ones(pos.size(0)), torch.zeros(neg.size(0))],
                                           dim=0).long().to(feats.device).unsqueeze(1)
                loss_cl += self.loss_const(sampled_feats, sampled_labels)

            loss_cl /= valid_num

        return {'loss_contrastive': loss_cl}

    def sample_pairs(self, feats, seg_label, cl_sampler='ori', batch_cl=True):
        B, C, H, W = feats.size()

        feats = feats.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        seg_label = seg_label.permute(0, 2, 3, 1).contiguous().view(B, -1)

        if batch_cl:
            pos_feats, neg_feats = self._sampling(feats, seg_label, cl_sampler)

        else:
            pos_feats, neg_feats = [], []
            for idx in range(feats.size()[0]):
                per_gt = seg_label[idx]
                if 1 in per_gt and 0 in per_gt:
                    per_pos_feats, per_neg_feats = self._sampling(feats[idx].unsuqeeze(0), per_gt.unsqueeze(0),
                                                                  cl_sampler)

                    pos_feats.append(per_pos_feats)
                    neg_feats.append(per_neg_feats)


        return pos_feats, neg_feats

    def _sampling(self, feats, seg_label, cl_sampler='ori'):
        """sample positive and negative pairs"""
        # sample positive pairs

        B, _, C = feats.size()


        feats = feats.view(-1, 1, C)
        seg_label = seg_label.view(-1)

        pos_idx = torch.nonzero(seg_label == 1, as_tuple=False)
        neg_idx = torch.nonzero(seg_label == 0, as_tuple=False)

        num_pos = pos_idx.size(0)
        num_neg = neg_idx.size(0)

        if cl_sampler == 'balance':

            pos_idx = pos_idx[torch.randperm(pos_idx.size(0))]
            neg_idx = neg_idx[torch.randperm(neg_idx.size(0))]

            min_points = min(num_pos, num_neg)
            if min_points < self.max_points:
                pos_idx_cur = pos_idx[:min_points]
                neg_idx_cur = neg_idx[:min_points]

                pos_idx_unused = pos_idx[min_points:]
                neg_idx_unused = neg_idx[min_points:]

            else:
                pos_idx_cur = pos_idx[:self.max_points]
                neg_idx_cur = neg_idx[:self.max_points]

                pos_idx_unused = pos_idx[self.max_points:]
                neg_idx_unused = neg_idx[self.max_points:]

        elif self.cl_sampler == 'ori':
            pos_idx = pos_idx[torch.randperm(pos_idx.size(0))]
            neg_idx = neg_idx[torch.randperm(neg_idx.size(0))]

            if num_pos < self.max_points:
                pos_idx_cur = pos_idx[:num_pos]
                neg_idx_cur = neg_idx[:(self.all_points - num_pos)]

                pos_idx_unused = pos_idx[num_pos:]
                neg_idx_unused = neg_idx[(self.all_points - num_pos):]

            elif num_neg < self.max_points:
                pos_idx_cur = pos_idx[:(self.all_points - num_neg)]
                neg_idx_cur = neg_idx[:num_neg]

                pos_idx_unused = pos_idx[(self.all_points - num_pos):]
                neg_idx_unused = neg_idx[num_neg:]

            else:
                pos_idx_cur = pos_idx[:self.max_points]
                neg_idx_cur = neg_idx[:self.max_points]

                pos_idx_unused = pos_idx[self.max_points:]
                neg_idx_unused = neg_idx[self.max_points:]

        elif self.cl_sampler == 'max':
            pos_idx = pos_idx[torch.randperm(pos_idx.size(0))]
            neg_idx = neg_idx[torch.randperm(neg_idx.size(0))]

            pos_idx_cur = pos_idx[:self.max_points]
            neg_idx_cur = neg_idx[:self.max_points]

            pos_idx_unused = pos_idx[self.max_points:]
            neg_idx_unused = neg_idx[self.max_points:]

        else:
            raise not NotImplementedError

        pos_feats = feats[pos_idx_cur].squeeze(1)
        neg_feats = feats[neg_idx_cur].squeeze(1)

        pos_feats = F.normalize(pos_feats, dim=2)
        neg_feats = F.normalize(neg_feats, dim=2)

        if self.use_memory:
            pos_feats, neg_feats = self.merge_memory(pos_feats, neg_feats)
            self.memory_bank.update(feats, pos_idx, pos_idx_unused, neg_idx, neg_idx_unused)

        return pos_feats, neg_feats


    def merge_memory(self, pos_feats, neg_feats):
        """merge samples extracted from memory bank with current samples"""
        pos_mem = self.memory_bank.sample_from_bank(num_sample=None, pos=True)
        neg_mem = self.memory_bank.sample_from_bank(num_sample=None, pos=False)

        if pos_mem is not None:
            pos_mem.requires_grad = True
            pos_feats = torch.cat([pos_feats, pos_mem], dim=0)
        if neg_mem is not None:
            neg_mem.requires_grad = True
            neg_feats = torch.cat([neg_feats, neg_mem], dim=0)

        return pos_feats, neg_feats


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
            seg_logits, _ = self.forward(inputs)

        return self.predict_by_feat(seg_logits, batch_img_metas)


    def save_deep_feature(self, feat, save_dir, batch_img_metas):
        """save intermediate feature map"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        feat = feat[0].detach().permute(1,2,0).cpu().numpy()
        img_meta = batch_img_metas[0]
        img_name = img_meta['img_path']
        img_name = os.path.basename(img_name).split('.')[0]
        img_name = img_name + '.npy'
        np.save(os.path.join(save_dir, img_name), feat)
        print('save feature map: ', img_name)


class MemoryBank(object):
    """
    Memory bank to store positive & negative features
        - update: update the memory bank
        -

    """
    def __init__(self, max_steps=3, max_sample=512, max_len=None):
        self.max_steps = max_steps
        self.max_sample = max_sample
        if max_len is None:
            self.max_len = max_steps * max_sample

        self.pos_bank = []
        self.neg_bank = []
        self.pos_valid = 0
        self.neg_valid = 0

        self.pos_flag = 0
        self.neg_flag = 0

        self.pos_num = 0
        self.neg_num = 0

        print('>>>> memory bank initialized <<<<')

    def reset_bank(self):
        self.pos_bank = []
        self.neg_bank = []

        self.pos_valid = 0
        self.neg_valid = 0

        self.pos_flag = 0
        self.neg_flag = 0


    def update(self, feats, pos_idx, pos_idx_unused, neg_idx, neg_idx_unused):
        """update the memory bank"""
        """
            1. forget the oldest memory if the memory bank is full
            2. add new memory
        """

        self.forget()
        feats = feats.detach()

        pos_idx = pos_idx.detach()
        pos_idx_unused = pos_idx_unused.detach()

        neg_idx = neg_idx.detach()
        neg_idx_unused = neg_idx_unused.detach()

        self.remember(feats, pos_idx, pos_idx_unused, pos=True)
        self.remember(feats, neg_idx, neg_idx_unused, pos=False)


    def remember(self, feats, idx, unused_idx, pos=True):

        if feats == None:
            memory = None
        else:
            feats = feats.detach()
            idx = idx.detach()
            unused_idx = unused_idx.detach()

            memory = self.exclusive_sampling(feats, idx, unused_idx)

        if pos:
            self.pos_bank.append(memory)
            num = memory.size(0)
            if num > 0:
                self.pos_valid += 1
            self.pos_flag += 1
            self.pos_num += num

        else:
            self.neg_bank.append(memory)
            num = memory.size(0)
            if memory.size(0) > 0:
                self.neg_valid += 1
            self.neg_flag += 1
            self.neg_num += num


    def forget(self):
        if self.pos_flag >= self.max_steps:
            out = self.pos_bank.pop(0)
            num = out.size(0)
            if num > 0:
                self.pos_valid -= 1

            self.pos_flag -= 1
            self.pos_num -= num

        if self.neg_flag >= self.max_steps:
            out = self.neg_bank.pop(0)
            num = out.size(0)
            if num > 0:
                self.neg_valid -= 1

            self.neg_flag -= 1
            self.neg_num -= num


    def exclusive_sampling(self, feats, idx, unused_idx):
        num_unused = unused_idx.size(0)
        num_used = idx.size(0)

        if num_unused >= self.max_sample:
            unused_idx = unused_idx[torch.randperm(num_unused)][:self.max_sample]
            samples = feats[unused_idx].squeeze(1)

        else:
            if num_unused > 0:
                samples_un = feats[idx].squeeze(1)
                num_sample_from_used = self.max_sample - num_unused
                cur_idx = idx[torch.randperm(num_used)][:num_sample_from_used]
                samples_cur = feats[cur_idx].squeeze(1)
                samples = torch.cat([samples_un, samples_cur], dim=0)

            else:
                samples = feats[idx].squeeze(1)

        samples = F.normalize(samples, dim=2)

        return samples


    def sample_from_bank(self, num_sample=None, pos=True):
        """
            sample from one memory bank
            if num_sample is None: sample all
            if sample is not None: sample num_sample
                - the latest memory first
                - unused memory first
        """

        if pos:
            bank = self.pos_bank
            valid_num = self.pos_valid
            memory_num = self.pos_num
        else:
            bank = self.neg_bank
            valid_num = self.pos_flag
            memory_num = self.neg_num

        # print('valid_num: ', valid_num)

        if valid_num == 0:
            return None
        else:
            memory = torch.cat(bank, dim=0)
            if num_sample is None:
                return memory
            else:
                if num_sample >= memory_num:
                    num_sample = memory_num
                sample = memory[-num_sample:]
                return sample


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

