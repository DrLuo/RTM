# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Sequence

# import mmcv
# from mmengine.fileio import FileClient
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample
# from mmseg.visualization import SegLocalVisualizer
from pathlib import Path
# from PIL import Image
import cv2
import numpy as np


@HOOKS.register_module()
class SegResultHook(Hook):
    """Segmentation Visualization Hook. Used to visualize validation and
    testing process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 draw: bool = False,
                 interval: int = 1,
                 save_dir = 'work_dirs',
                 save_mask: bool = False,
                 save_prob: bool = False,
                 class_idx: int = 1,
                 binary: bool = False,
                 use_sigmoid: bool = False

                 ):

        self.save_dir = save_dir
        self.interval = interval
        self.save_mask = save_mask
        self.save_prob = save_prob
        self.class_idx = class_idx
        self.binary = binary
        self.use_sigmoid = use_sigmoid


        self.file_client = None
        self.draw = draw
        if not self.draw:
            warnings.warn('The draw is False, it means that the '
                          'hook for result visualization will not take '
                          'effect. The results will NOT be '
                          'visualized or stored.')

        if self.save_mask:
            self.mask_dir = osp.join(self.save_dir, 'pred_mask')
            Path(self.mask_dir).mkdir(parents=True, exist_ok=True)

        if self.save_prob:
            self.prob_dir = osp.join(self.save_dir, 'pred_heatmap')
            Path(self.prob_dir).mkdir(parents=True, exist_ok=True)


    def after_test_iter(self,
                        runner: Runner,
                        batch_idx: int,
                        data_batch: dict,
                        outputs: Sequence[SegDataSample]) -> None:
        if self.draw is False:
            return

        if self.every_n_inner_iters(batch_idx, self.interval):
            for output in outputs:
                img_path = output.img_path
                save_name = osp.basename(img_path)
                save_name = osp.splitext(save_name)[0] + '.png'

                if self.save_prob:
                    heat_map_data = output.get('seg_logits')
                    heat_map = heat_map_data.get('data')

                    if self.use_sigmoid:

                        heat_map = heat_map.sigmoid()
                        heat_map = heat_map.cpu().numpy()
                        heat_map = heat_map[0]
                    else:
                        heat_map = heat_map.softmax(dim=0)

                        heat_map = heat_map.cpu().numpy()
                        heat_map = heat_map[self.class_idx]

                    heat_map = heat_map * 255
                    heat_map = heat_map.round()

                    cv2.imwrite(osp.join(self.prob_dir, save_name), heat_map.astype(np.uint8))


                if self.save_mask:

                    pred_mask_data = output.get('pred_sem_seg')
                    pred_mask = pred_mask_data.numpy().get('data').squeeze()

                    if self.binary:
                        pred_mask = pred_mask * 255

                    cv2.imwrite(osp.join(self.mask_dir, save_name), pred_mask.astype(np.uint8))
