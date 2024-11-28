# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.nn as nn
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,
                      build_activation_layer, build_norm_layer)
from mmengine.model import BaseModule

from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)

from mmseg.registry import MODELS
from ..utils import resize


@MODELS.register_module()
class WrappedEncoder(BaseModule):
    """Dual Stream Encoder for Semantic Segmentation.

    """

    def __init__(self,
                 backbone,
                 preprocessor=None,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)

        self.with_fore_fusion = False
        self.with_post_fusion = False

        self.backbone = MODELS.build(backbone)
        self._init_preprocessor(preprocessor)

    def _init_preprocessor(self, preprocessor: ConfigType) -> None:
        """Initialize ``preprocessor_sec``"""
        if preprocessor is not None:
            self.preprocessor = MODELS.build(preprocessor)


    def forward(self, x):
        #  stole refactoring code from Coin Cheung, thanks
        x = self.preprocessor(x)
        x = self.backbone(x)

        return tuple(x)



    @property
    def with_preprocessor(self) -> bool:
        """bool: whether the segmentor has neck"""
        return hasattr(self, 'preprocessor') and self.preprocessor is not None

