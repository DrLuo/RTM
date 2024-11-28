# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor

from pathlib import Path
import os
import numpy as np
import cv2


@MODELS.register_module()
class MyModelFull(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    """

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 backbone_sec: OptConfigType = None,
                 preprocessor: OptConfigType = None,
                 preprocessor_sec: [OptConfigType, dict] = None,   # preprocess before the second backbone
                 key_sec: Optional = None,
                 pack_type = 'list',
                 fuser: OptConfigType = None,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 merge_input: bool = False,
                 extra_head: OptConfigType = None,
                 use_extra: bool = False,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 pretrained_sec: Optional[str] = None,
                 vis_preprocessor: str = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.use_extra = use_extra
        self.key_sec = key_sec
        self.merge_input = merge_input
        self.pack_type = pack_type
        self.vis_preprocessor = vis_preprocessor

        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)

        # init second feature extractor
        if backbone_sec is not None:
            if pretrained_sec is not None:
                assert backbone_sec.get('pretrained') is None, \
                    'both backbone 2 and segmentor set pretrained weight'
                backbone.pretrained = pretrained_sec
            self.backbone_sec = MODELS.build(backbone_sec)


        if neck is not None:
            self.neck = MODELS.build(neck)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)


        self._init_extra_head(extra_head)
        self._init_fuser(fuser)
        self._init_preprocessor(preprocessor)
        self._init_preprocessor_sec(preprocessor_sec)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head


    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_fuser(self, fuser: ConfigType) -> None:
        """Initialize ``fuser``"""
        if fuser is not None:
            self.fuser = MODELS.build(fuser)
            # self.fuse_mode = self.fuser.fuse_mode

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def _init_extra_head(self, extra_head: ConfigType) -> None:
        """Initialize ``extra_head``, can be supervised by extra info"""
        if extra_head is not None:
            if isinstance(extra_head, list):
                self.extra_head = nn.ModuleList()
                for head_cfg in extra_head:
                    self.extra_head.append(MODELS.build(head_cfg))
            else:
                self.extra_head = MODELS.build(extra_head)

    def _init_preprocessor(self, preprocessor: ConfigType) -> None:
        """Initialize ``preprocessor``"""
        if preprocessor is not None:
            self.preprocessor = MODELS.build(preprocessor)
            if self.backbone is None:
                warnings.warn('preprocessor defined but backbone is UNDEFINED')


    def _init_preprocessor_sec(self, preprocessor_sec: ConfigType) -> None:
        """Initialize ``preprocessor_sec``"""
        if preprocessor_sec is not None:
            if isinstance(preprocessor_sec, list):
                self.preprocessor_sec = nn.ModuleList()
                self.extra_names = []
                for cfg_name in preprocessor_sec:
                    extra_name = cfg_name[0]
                    self.preprocessor_sec.append(MODELS.build(cfg_name[1]))
                    self.extra_names.append(extra_name)
            else:
                self.preprocessor_sec = MODELS.build(preprocessor_sec)
                if not self.with_backbone_sec:
                    warnings.warn('preprocessor_sec defined but backbone second is UNDEFINED')


    # TODO: modify forward
    def extract_feat(self, inputs) -> List[Tensor]:
        """Extract features from image"""
        x = self.backbone(inputs)

        return x

    def extract_feat_sec(self, inputs) -> List[Tensor]:
        """Extract features from image"""
        x = self.backbone_sec(inputs)

        return x



    def forward_neck(self, x) -> List[Tensor]:
        return self.neck(x)

    def forward_encoder(self, inputs, extras):

        # preprocessing
        if self.with_preprocessor:
            inputs = self.preprocessor(inputs)

        if isinstance(self.key_sec, str):
            if self.key_sec == 'img':
                y = inputs
            else:
                y = extras[self.key_sec]
        else:
            y = extras

        if self.with_preprocessor_sec:
            if isinstance(self.preprocessor_sec, nn.ModuleList):
                # if self.multi_preprocessor:
                y = []
                for i in range(len(self.extra_names)):
                    extra_name = self.extra_names[i]
                    if extra_name == 'img':
                        y_i = inputs
                    elif extra_name == 'all':
                        y_i = extras
                    else:
                        y_i = extras[extra_name]

                    y.append(self.preprocessor_sec[i](y_i))
            elif isinstance(self.preprocessor_sec, nn.Module):
                y = self.preprocessor_sec(y)
            else:
                raise NotImplementedError

        if self.vis_preprocessor is not None:
            Path(self.vis_preprocessor).mkdir(parents=True, exist_ok=True)
            if isinstance(self.preprocessor_sec, nn.ModuleList):
                for i in range(len(self.extra_names)):
                    extra_name = self.extra_names[i]
                    img_name = os.path.basename(extras['img_path'][0])
                    filename = os.path.join(self.vis_preprocessor, img_name + f'_{extra_name}.png')
                    vis_img = y[i].detach().cpu().numpy().transpose(1, 2, 0)
                    vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min())
                    vis_img = (vis_img * 255).astype(np.uint8)
                    cv2.imwrite(filename, vis_img)
            else:
                img_name = os.path.basename(extras['img_path'][0])
                filename = os.path.join(self.vis_preprocessor, img_name + f'_{self.key_sec}.png')
                vis_img = y.detach().cpu().numpy().transpose(1, 2, 0)
                vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min())
                vis_img = (vis_img * 255).astype(np.uint8)
                cv2.imwrite(filename, vis_img)


        if self.merge_input:
            inputs = [inputs, y]

        x = self.extract_feat(inputs)
        if self.with_backbone_sec:

            y = self.extract_feat_sec(y)

            x = self.fuser(x, y)

        if self.with_neck:
            x = self.neck(x)
        return x


    # TODO: modify forward in inference
    def encode_decode(self, inputs: Tensor, extras,
                      batch_img_metas: List[dict]) -> List[Tensor]:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.forward_encoder(inputs, extras)

        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)

        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor], extras,
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        if self.use_extra:
            assert extras is not None, 'need extra feature'
            loss_decode = self.decode_head.loss(inputs, extras, data_samples,
                                                self.train_cfg)
        else:
            loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def _extra_head_forward_train(self, inputs: List[Tensor], extras) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.extra_head, nn.ModuleList):
            for idx, extra_head in enumerate(self.extra_head):
                loss_extra = extra_head.loss(inputs, extras, self.train_cfg)
                losses.update(add_prefix(loss_extra, f'extra_{idx}'))
        else:
            loss_extra = self.extra_head.loss(inputs, extras,
                                              self.train_cfg)
            losses.update(add_prefix(loss_extra, 'extra'))

        return losses



    # TODO: add freq
    def loss(self, inputs: Tensor, extras, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.forward_encoder(inputs, extras)

        losses = dict()


        loss_decode = self._decode_head_forward_train(x, extras, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        if self.with_extra_head:
            loss_extra = self._extra_head_forward_train(x, extras)
            losses.update(loss_extra)

        return losses

    # TODO:
    def predict(self,
                inputs: Tensor, extras,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """


        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, extras, batch_img_metas)


        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor, extras,# dcts: Tensor, qtables: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """

        x = self.forward_encoder(inputs, extras)

        return self.decode_head.forward(x)


    def forward(self,
                inputs: Tensor, extras,
                data_samples: OptSampleList = None,
                mode: str = 'tensor'):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`SegDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape (N, C, ...) in
                general.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """


        if mode == 'loss':
            return self.loss(inputs, extras, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, extras, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, extras, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')


    # TODO: Crop DCT during inference
    def slide_inference(self, inputs: Tensor, extras,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_extra = dict()
                for k in extras:
                    if k == 'qtable' or k == 'edge' or k == 'dis_map':
                        crop_extra[k] = extras[k]
                    else:
                        item = extras[k]
                        if len(item.size()) == 4:
                            crop_extra[k] = item[:, :, y1:y2, x1:x2]
                        elif len(item.size()) == 3:
                            crop_extra[k] = item[:, y1:y2, x1:x2]

                crop_seg_logit = self.encode_decode(crop_img, crop_extra, batch_img_metas)  #TODO
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor, extras,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, extras, batch_img_metas)    # TODO

        return seg_logits

    def inference(self, inputs: Tensor, extras, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = batch_img_metas[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in batch_img_metas)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, extras, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, extras, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    @property
    def with_extra_head(self) -> bool:
        """bool: whether the segmentor has neck"""
        return hasattr(self, 'extra_head') and self.extra_head is not None

    @property
    def with_backbone_sec(self) -> bool:
        """bool: whether the segmentor has neck"""
        return hasattr(self, 'backbone_sec') and self.backbone_sec is not None

    @property
    def with_fuser(self) -> bool:
        """bool: whether the segmentor has neck"""
        return hasattr(self, 'fuser') and self.fuser is not None

    @property
    def with_preprocessor(self) -> bool:
        """bool: whether the segmentor has neck"""
        return hasattr(self, 'preprocessor') and self.preprocessor is not None

    @property
    def with_preprocessor_sec(self) -> bool:
        """bool: whether the segmentor has neck"""
        return hasattr(self, 'preprocessor_sec') and self.preprocessor_sec is not None

    @property
    def multi_preprocessor(self) -> bool:
        """bool: whether the segmentor has neck"""
        return hasattr(self, 'extra_names') and self.extra_names is not None



