# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union, Iterable
from mmcv.image.geometric import _scale_size

import mmengine
import torch
from torch.nn import functional as F

import cv2
from PIL import Image
import mmcv
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmengine.utils import is_tuple_of
from numpy import random
from scipy.ndimage import gaussian_filter

from mmseg.datasets.dataset_wrappers import MultiImageMixDataset
from mmseg.registry import TRANSFORMS

# from mmengine.runner import
Number = Union[int, float]
import albumentations as A

try:
    import tempfile
    import jpegio
except:
    tempfile = None
    jpegio = None


@TRANSFORMS.register_module()
class ResizeToMultiple(BaseTransform):
    """Resize images & seg to multiple of divisor.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - img_shape
    - pad_shape

    Args:
        size_divisor (int): images and gt seg maps need to resize to multiple
            of size_divisor. Default: 32.
        interpolation (str, optional): The interpolation mode of image resize.
            Default: None
    """

    def __init__(self, size_divisor=32, interpolation=None):
        self.size_divisor = size_divisor
        self.interpolation = interpolation

    def transform(self, results: dict) -> dict:
        """Call function to resize images, semantic segmentation map to
        multiple of size divisor.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape' keys are updated.
        """
        # Align image to multiple of size divisor.
        img = results['img']
        img = mmcv.imresize_to_multiple(
            img,
            self.size_divisor,
            scale_factor=1,
            interpolation=self.interpolation
            if self.interpolation else 'bilinear')

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['pad_shape'] = img.shape[:2]

        # Align segmentation map to multiple of size divisor.
        for key in results.get('seg_fields', []):
            gt_seg = results[key]
            gt_seg = mmcv.imresize_to_multiple(
                gt_seg,
                self.size_divisor,
                scale_factor=1,
                interpolation='nearest')
            results[key] = gt_seg

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(size_divisor={self.size_divisor}, '
                     f'interpolation={self.interpolation})')
        return repr_str


@TRANSFORMS.register_module()
class Rerange(BaseTransform):
    """Rerange the image pixel value.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        min_value (float or int): Minimum value of the reranged image.
            Default: 0.
        max_value (float or int): Maximum value of the reranged image.
            Default: 255.
    """

    def __init__(self, min_value=0, max_value=255):
        assert isinstance(min_value, float) or isinstance(min_value, int)
        assert isinstance(max_value, float) or isinstance(max_value, int)
        assert min_value < max_value
        self.min_value = min_value
        self.max_value = max_value

    def transform(self, results: dict) -> dict:
        """Call function to rerange images.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Reranged results.
        """

        img = results['img']
        img_min_value = np.min(img)
        img_max_value = np.max(img)

        assert img_min_value < img_max_value
        # rerange to [0, 1]
        img = (img - img_min_value) / (img_max_value - img_min_value)
        # rerange to [min_value, max_value]
        img = img * (self.max_value - self.min_value) + self.min_value
        results['img'] = img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_value={self.min_value}, max_value={self.max_value})'
        return repr_str


@TRANSFORMS.register_module()
class CLAHE(BaseTransform):
    """Use CLAHE method to process the image.

    See `ZUIDERVELD,K. Contrast Limited Adaptive Histogram Equalization[J].
    Graphics Gems, 1994:474-485.` for more information.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        clip_limit (float): Threshold for contrast limiting. Default: 40.0.
        tile_grid_size (tuple[int]): Size of grid for histogram equalization.
            Input image will be divided into equally sized rectangular tiles.
            It defines the number of tiles in row and column. Default: (8, 8).
    """

    def __init__(self, clip_limit=40.0, tile_grid_size=(8, 8)):
        assert isinstance(clip_limit, (float, int))
        self.clip_limit = clip_limit
        assert is_tuple_of(tile_grid_size, int)
        assert len(tile_grid_size) == 2
        self.tile_grid_size = tile_grid_size

    def transform(self, results: dict) -> dict:
        """Call function to Use CLAHE method process images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """

        for i in range(results['img'].shape[2]):
            results['img'][:, :, i] = mmcv.clahe(
                np.array(results['img'][:, :, i], dtype=np.uint8),
                self.clip_limit, self.tile_grid_size)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(clip_limit={self.clip_limit}, '\
                    f'tile_grid_size={self.tile_grid_size})'
        return repr_str


@TRANSFORMS.register_module()
class RandomCrop(BaseTransform):
    """Random crop the image & seg.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - img_shape
    - gt_seg_map


    Args:
        crop_size (Union[int, Tuple[int, int]]):  Expected size after cropping
            with the format of (h, w). If set to an integer, then cropping
            width and height are equal to this integer.
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
        ignore_index (int): The label index to be ignored. Default: 255
    """

    def __init__(self,
                 crop_size: Union[int, Tuple[int, int]],
                 cat_max_ratio: float = 1.,
                 ignore_index: int = 255):
        super().__init__()
        assert isinstance(crop_size, int) or (
            isinstance(crop_size, tuple) and len(crop_size) == 2
        ), 'The expected crop_size is an integer, or a tuple containing two '
        'intergers'

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    @cache_randomness
    def crop_bbox(self, results: dict) -> tuple:
        """get a crop bounding box.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            tuple: Coordinates of the cropped image.
        """

        def generate_crop_bbox(img: np.ndarray) -> tuple:
            """Randomly get a crop bounding box.

            Args:
                img (np.ndarray): Original input image.

            Returns:
                tuple: Coordinates of the cropped image.
            """

            margin_h = max(img.shape[0] - self.crop_size[0], 0)
            margin_w = max(img.shape[1] - self.crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            return crop_y1, crop_y2, crop_x1, crop_x2

        img = results['img']
        crop_bbox = generate_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_seg_map'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = generate_crop_bbox(img)

        return crop_bbox

    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop from ``img``

        Args:
            img (np.ndarray): Original input image.
            crop_bbox (tuple): Coordinates of the cropped image.

        Returns:
            np.ndarray: The cropped image.
        """

        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        crop_bbox = self.crop_bbox(results)

        # crop the image
        img = self.crop(img, crop_bbox)

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@TRANSFORMS.register_module()
class RandomRotate(BaseTransform):
    """Rotate the image & seg.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - gt_seg_map

    Args:
        prob (float): The rotation probability.
        degree (float, tuple[float]): Range of degrees to select from. If
            degree is a number instead of tuple like (min, max),
            the range of degree will be (``-degree``, ``+degree``)
        pad_val (float, optional): Padding value of image. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used. Default: None.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image. Default: False
    """

    def __init__(self,
                 prob,
                 degree,
                 pad_val=0,
                 seg_pad_val=255,
                 center=None,
                 auto_bound=False):
        self.prob = prob
        assert prob >= 0 and prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'
        self.pal_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.center = center
        self.auto_bound = auto_bound

    @cache_randomness
    def generate_degree(self):
        return np.random.rand() < self.prob, np.random.uniform(
            min(*self.degree), max(*self.degree))

    def transform(self, results: dict) -> dict:
        """Call function to rotate image, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """

        rotate, degree = self.generate_degree()
        if rotate:
            # rotate image
            results['img'] = mmcv.imrotate(
                results['img'],
                angle=degree,
                border_value=self.pal_val,
                center=self.center,
                auto_bound=self.auto_bound)

            # rotate segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imrotate(
                    results[key],
                    angle=degree,
                    border_value=self.seg_pad_val,
                    center=self.center,
                    auto_bound=self.auto_bound,
                    interpolation='nearest')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, ' \
                    f'degree={self.degree}, ' \
                    f'pad_val={self.pal_val}, ' \
                    f'seg_pad_val={self.seg_pad_val}, ' \
                    f'center={self.center}, ' \
                    f'auto_bound={self.auto_bound})'
        return repr_str


@TRANSFORMS.register_module()
class RGB2Gray(BaseTransform):
    """Convert RGB image to grayscale image.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_shape

    This transform calculate the weighted mean of input image channels with
    ``weights`` and then expand the channels to ``out_channels``. When
    ``out_channels`` is None, the number of output channels is the same as
    input channels.

    Args:
        out_channels (int): Expected number of output channels after
            transforming. Default: None.
        weights (tuple[float]): The weights to calculate the weighted mean.
            Default: (0.299, 0.587, 0.114).
    """

    def __init__(self, out_channels=None, weights=(0.299, 0.587, 0.114)):
        assert out_channels is None or out_channels > 0
        self.out_channels = out_channels
        assert isinstance(weights, tuple)
        for item in weights:
            assert isinstance(item, (float, int))
        self.weights = weights

    def transform(self, results: dict) -> dict:
        """Call function to convert RGB image to grayscale image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with grayscale image.
        """
        img = results['img']
        assert len(img.shape) == 3
        assert img.shape[2] == len(self.weights)
        weights = np.array(self.weights).reshape((1, 1, -1))
        img = (img * weights).sum(2, keepdims=True)
        if self.out_channels is None:
            img = img.repeat(weights.shape[2], axis=2)
        else:
            img = img.repeat(self.out_channels, axis=2)

        results['img'] = img
        results['img_shape'] = img.shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(out_channels={self.out_channels}, ' \
                    f'weights={self.weights})'
        return repr_str


@TRANSFORMS.register_module()
class AdjustGamma(BaseTransform):
    """Using gamma correction to process the image.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        gamma (float or int): Gamma value used in gamma correction.
            Default: 1.0.
    """

    def __init__(self, gamma=1.0):
        assert isinstance(gamma, float) or isinstance(gamma, int)
        assert gamma > 0
        self.gamma = gamma
        inv_gamma = 1.0 / gamma
        self.table = np.array([(i / 255.0)**inv_gamma * 255
                               for i in np.arange(256)]).astype('uint8')

    def transform(self, results: dict) -> dict:
        """Call function to process the image with gamma correction.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """

        results['img'] = mmcv.lut_transform(
            np.array(results['img'], dtype=np.uint8), self.table)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(gamma={self.gamma})'


@TRANSFORMS.register_module()
class SegRescale(BaseTransform):
    """Rescale semantic segmentation maps.

    Required Keys:

    - gt_seg_map

    Modified Keys:

    - gt_seg_map

    Args:
        scale_factor (float): The scale factor of the final output.
    """

    def __init__(self, scale_factor=1):
        self.scale_factor = scale_factor

    def transform(self, results: dict) -> dict:
        """Call function to scale the semantic segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with semantic segmentation map scaled.
        """
        for key in results.get('seg_fields', []):
            if self.scale_factor != 1:
                results[key] = mmcv.imrescale(
                    results[key], self.scale_factor, interpolation='nearest')
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(scale_factor={self.scale_factor})'


@TRANSFORMS.register_module()
class PhotoMetricDistortion(BaseTransform):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: Sequence[float] = (0.5, 1.5),
                 saturation_range: Sequence[float] = (0.5, 1.5),
                 hue_delta: int = 18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self,
                img: np.ndarray,
                alpha: int = 1,
                beta: int = 0) -> np.ndarray:
        """Multiple with alpha and add beat with clip.

        Args:
            img (np.ndarray): The input image.
            alpha (int): Image weights, change the contrast/saturation
                of the image. Default: 1
            beta (int): Image bias, change the brightness of the
                image. Default: 0

        Returns:
            np.ndarray: The transformed image.
        """

        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img: np.ndarray) -> np.ndarray:
        """Brightness distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after brightness change.
        """

        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img: np.ndarray) -> np.ndarray:
        """Contrast distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after contrast change.
        """

        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img: np.ndarray) -> np.ndarray:
        """Saturation distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after saturation change.
        """

        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img: np.ndarray) -> np.ndarray:
        """Hue distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after hue change.
        """

        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        img = results['img']
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str


@TRANSFORMS.register_module()
class RandomCutOut(BaseTransform):
    """CutOut operation.

    Randomly drop some regions of image used in
    `Cutout <https://arxiv.org/abs/1708.04552>`_.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - gt_seg_map

    Args:
        prob (float): cutout probability.
        n_holes (int | tuple[int, int]): Number of regions to be dropped.
            If it is given as a list, number of holes will be randomly
            selected from the closed interval [`n_holes[0]`, `n_holes[1]`].
        cutout_shape (tuple[int, int] | list[tuple[int, int]]): The candidate
            shape of dropped regions. It can be `tuple[int, int]` to use a
            fixed cutout shape, or `list[tuple[int, int]]` to randomly choose
            shape from the list.
        cutout_ratio (tuple[float, float] | list[tuple[float, float]]): The
            candidate ratio of dropped regions. It can be `tuple[float, float]`
            to use a fixed ratio or `list[tuple[float, float]]` to randomly
            choose ratio from the list. Please note that `cutout_shape`
            and `cutout_ratio` cannot be both given at the same time.
        fill_in (tuple[float, float, float] | tuple[int, int, int]): The value
            of pixel to fill in the dropped regions. Default: (0, 0, 0).
        seg_fill_in (int): The labels of pixel to fill in the dropped regions.
            If seg_fill_in is None, skip. Default: None.
    """

    def __init__(self,
                 prob,
                 n_holes,
                 cutout_shape=None,
                 cutout_ratio=None,
                 fill_in=(0, 0, 0),
                 seg_fill_in=None):

        assert 0 <= prob and prob <= 1
        assert (cutout_shape is None) ^ (cutout_ratio is None), \
            'Either cutout_shape or cutout_ratio should be specified.'
        assert (isinstance(cutout_shape, (list, tuple))
                or isinstance(cutout_ratio, (list, tuple)))
        if isinstance(n_holes, tuple):
            assert len(n_holes) == 2 and 0 <= n_holes[0] < n_holes[1]
        else:
            n_holes = (n_holes, n_holes)
        if seg_fill_in is not None:
            assert (isinstance(seg_fill_in, int) and 0 <= seg_fill_in
                    and seg_fill_in <= 255)
        self.prob = prob
        self.n_holes = n_holes
        self.fill_in = fill_in
        self.seg_fill_in = seg_fill_in
        self.with_ratio = cutout_ratio is not None
        self.candidates = cutout_ratio if self.with_ratio else cutout_shape
        if not isinstance(self.candidates, list):
            self.candidates = [self.candidates]

    @cache_randomness
    def do_cutout(self):
        return np.random.rand() < self.prob

    @cache_randomness
    def generate_patches(self, results):
        cutout = self.do_cutout()

        h, w, _ = results['img'].shape
        if cutout:
            n_holes = np.random.randint(self.n_holes[0], self.n_holes[1] + 1)
        else:
            n_holes = 0
        x1_lst = []
        y1_lst = []
        index_lst = []
        for _ in range(n_holes):
            x1_lst.append(np.random.randint(0, w))
            y1_lst.append(np.random.randint(0, h))
            index_lst.append(np.random.randint(0, len(self.candidates)))
        return cutout, n_holes, x1_lst, y1_lst, index_lst

    def transform(self, results: dict) -> dict:
        """Call function to drop some regions of image."""
        cutout, n_holes, x1_lst, y1_lst, index_lst = self.generate_patches(
            results)
        if cutout:
            h, w, c = results['img'].shape
            for i in range(n_holes):
                x1 = x1_lst[i]
                y1 = y1_lst[i]
                index = index_lst[i]
                if not self.with_ratio:
                    cutout_w, cutout_h = self.candidates[index]
                else:
                    cutout_w = int(self.candidates[index][0] * w)
                    cutout_h = int(self.candidates[index][1] * h)

                x2 = np.clip(x1 + cutout_w, 0, w)
                y2 = np.clip(y1 + cutout_h, 0, h)
                results['img'][y1:y2, x1:x2, :] = self.fill_in

                if self.seg_fill_in is not None:
                    for key in results.get('seg_fields', []):
                        results[key][y1:y2, x1:x2] = self.seg_fill_in

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'n_holes={self.n_holes}, '
        repr_str += (f'cutout_ratio={self.candidates}, ' if self.with_ratio
                     else f'cutout_shape={self.candidates}, ')
        repr_str += f'fill_in={self.fill_in}, '
        repr_str += f'seg_fill_in={self.seg_fill_in})'
        return repr_str


@TRANSFORMS.register_module()
class RandomRotFlip(BaseTransform):
    """Rotate and flip the image & seg or just rotate the image & seg.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - gt_seg_map

    Args:
        rotate_prob (float): The probability of rotate image.
        flip_prob (float): The probability of rotate&flip image.
        degree (float, tuple[float]): Range of degrees to select from. If
            degree is a number instead of tuple like (min, max),
            the range of degree will be (``-degree``, ``+degree``)
    """

    def __init__(self, rotate_prob=0.5, flip_prob=0.5, degree=(-20, 20)):
        self.rotate_prob = rotate_prob
        self.flip_prob = flip_prob
        assert 0 <= rotate_prob <= 1 and 0 <= flip_prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'

    def random_rot_flip(self, results: dict) -> dict:
        k = np.random.randint(0, 4)
        results['img'] = np.rot90(results['img'], k)
        for key in results.get('seg_fields', []):
            results[key] = np.rot90(results[key], k)
        axis = np.random.randint(0, 2)
        results['img'] = np.flip(results['img'], axis=axis).copy()
        for key in results.get('seg_fields', []):
            results[key] = np.flip(results[key], axis=axis).copy()
        return results

    def random_rotate(self, results: dict) -> dict:
        angle = np.random.uniform(min(*self.degree), max(*self.degree))
        results['img'] = mmcv.imrotate(results['img'], angle=angle)
        for key in results.get('seg_fields', []):
            results[key] = mmcv.imrotate(results[key], angle=angle)
        return results

    def transform(self, results: dict) -> dict:
        """Call function to rotate or rotate & flip image, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated or rotated & flipped results.
        """
        rotate_flag = 0
        if random.random() < self.rotate_prob:
            results = self.random_rotate(results)
            rotate_flag = 1
        if random.random() < self.flip_prob and rotate_flag == 0:
            results = self.random_rot_flip(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(rotate_prob={self.rotate_prob}, ' \
                    f'flip_prob={self.flip_prob}, ' \
                    f'degree={self.degree})'
        return repr_str


@TRANSFORMS.register_module()
class RandomMosaic(BaseTransform):
    """Mosaic augmentation. Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.

    .. code:: text

                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |  pad      |
                |      +-----------+           |
                |      |           |           |
                |      |  image1   |--------+  |
                |      |           |        |  |
                |      |           | image2 |  |
     center_y   |----+-------------+-----------|
                |    |   cropped   |           |
                |pad |   image3    |  image4   |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     The mosaic transform steps are as follows:
         1. Choose the mosaic center as the intersections of 4 images
         2. Get the left top image according to the index, and randomly
            sample another 3 images from the custom dataset.
         3. Sub image will be cropped if image is larger than mosaic patch

    Required Keys:

    - img
    - gt_seg_map
    - mix_results

    Modified Keys:

    - img
    - img_shape
    - ori_shape
    - gt_seg_map

    Args:
        prob (float): mosaic probability.
        img_scale (Sequence[int]): Image size after mosaic pipeline of
            a single image. The size of the output image is four times
            that of a single image. The output image comprises 4 single images.
            Default: (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Default: (0.5, 1.5).
        pad_val (int): Pad value. Default: 0.
        seg_pad_val (int): Pad value of segmentation map. Default: 255.
    """

    def __init__(self,
                 prob,
                 img_scale=(640, 640),
                 center_ratio_range=(0.5, 1.5),
                 pad_val=0,
                 seg_pad_val=255):
        assert 0 <= prob and prob <= 1
        assert isinstance(img_scale, tuple)
        self.prob = prob
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    @cache_randomness
    def do_mosaic(self):
        return np.random.rand() < self.prob

    def transform(self, results: dict) -> dict:
        """Call function to make a mosaic of image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with mosaic transformed.
        """
        mosaic = self.do_mosaic()
        if mosaic:
            results = self._mosaic_transform_img(results)
            results = self._mosaic_transform_seg(results)
        return results

    def get_indices(self, dataset: MultiImageMixDataset) -> list:
        """Call function to collect indices.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indices.
        """

        indices = [random.randint(0, len(dataset)) for _ in range(3)]
        return indices

    @cache_randomness
    def generate_mosaic_center(self):
        # mosaic center x, y
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        return center_x, center_y

    def _mosaic_transform_img(self, results: dict) -> dict:
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        if len(results['img'].shape) == 3:
            c = results['img'].shape[2]
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), c),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)

        # mosaic center x, y
        self.center_x, self.center_y = self.generate_mosaic_center()
        center_position = (self.center_x, self.center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                result_patch = copy.deepcopy(results)
            else:
                result_patch = copy.deepcopy(results['mix_results'][i - 1])

            img_i = result_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[0] / h_i,
                                self.img_scale[1] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['ori_shape'] = mosaic_img.shape

        return results

    def _mosaic_transform_seg(self, results: dict) -> dict:
        """Mosaic transform function for label annotations.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        for key in results.get('seg_fields', []):
            mosaic_seg = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.seg_pad_val,
                dtype=results[key].dtype)

            # mosaic center x, y
            center_position = (self.center_x, self.center_y)

            loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
            for i, loc in enumerate(loc_strs):
                if loc == 'top_left':
                    result_patch = copy.deepcopy(results)
                else:
                    result_patch = copy.deepcopy(results['mix_results'][i - 1])

                gt_seg_i = result_patch[key]
                h_i, w_i = gt_seg_i.shape[:2]
                # keep_ratio resize
                scale_ratio_i = min(self.img_scale[0] / h_i,
                                    self.img_scale[1] / w_i)
                gt_seg_i = mmcv.imresize(
                    gt_seg_i,
                    (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)),
                    interpolation='nearest')

                # compute the combine parameters
                paste_coord, crop_coord = self._mosaic_combine(
                    loc, center_position, gt_seg_i.shape[:2][::-1])
                x1_p, y1_p, x2_p, y2_p = paste_coord
                x1_c, y1_c, x2_c, y2_c = crop_coord

                # crop and paste image
                mosaic_seg[y1_p:y2_p, x1_p:x2_p] = gt_seg_i[y1_c:y2_c,
                                                            x1_c:x2_c]

            results[key] = mosaic_seg

        return results

    def _mosaic_combine(self, loc: str, center_position_xy: Sequence[float],
                        img_shape_wh: Sequence[int]) -> tuple:
        """Calculate global coordinate of mosaic image and local coordinate of
        cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ('top_left',
              'top_right', 'bottom_left', 'bottom_right').
            center_position_xy (Sequence[float]): Mixing center for 4 images,
                (x, y).
            img_shape_wh (Sequence[int]): Width and height of sub-image

        Returns:
            tuple[tuple[float]]: Corresponding coordinate of pasting and
                cropping
                - paste_coord (tuple): paste corner coordinate in mosaic image.
                - crop_coord (tuple): crop corner coordinate in mosaic image.
        """

        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        if loc == 'top_left':
            # index0 to top left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             center_position_xy[0], \
                             center_position_xy[1]
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (
                y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == 'top_right':
            # index1 to top right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[1] * 2), \
                             center_position_xy[1]
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
                img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == 'bottom_left':
            # index2 to bottom left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             center_position_xy[1], \
                             center_position_xy[0], \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
                y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             center_position_xy[1], \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[1] * 2), \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = 0, 0, min(img_shape_wh[0],
                                   x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'seg_pad_val={self.pad_val})'
        return repr_str


@TRANSFORMS.register_module()
class GenerateEdge(BaseTransform):
    """Generate Edge for CE2P approach.

    Edge will be used to calculate loss of
    `CE2P <https://arxiv.org/abs/1809.05996>`_.

    Modified from https://github.com/liutinglt/CE2P/blob/master/dataset/target_generation.py # noqa:E501

    Required Keys:

        - img_shape
        - gt_seg_map

    Added Keys:
        - gt_edge_map (np.ndarray, uint8): The edge annotation generated from the
            seg map by extracting border between different semantics.

    Args:
        edge_width (int): The width of edge. Default to 3.
        ignore_index (int): Index that will be ignored. Default to 255.
    """

    def __init__(self, edge_width: int = 3, ignore_index: int = 255) -> None:
        super().__init__()
        self.edge_width = edge_width
        self.ignore_index = ignore_index

    def transform(self, results: Dict) -> Dict:
        """Call function to generate edge from segmentation map.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with edge mask.
        """
        h, w = results['img_shape']
        edge = np.zeros((h, w), dtype=np.uint8)
        seg_map = results['gt_seg_map']

        # down
        edge_down = edge[1:h, :]
        edge_down[(seg_map[1:h, :] != seg_map[:h - 1, :])
                  & (seg_map[1:h, :] != self.ignore_index) &
                  (seg_map[:h - 1, :] != self.ignore_index)] = 1
        # left
        edge_left = edge[:, :w - 1]
        edge_left[(seg_map[:, :w - 1] != seg_map[:, 1:w])
                  & (seg_map[:, :w - 1] != self.ignore_index) &
                  (seg_map[:, 1:w] != self.ignore_index)] = 1
        # up_left
        edge_upleft = edge[:h - 1, :w - 1]
        edge_upleft[(seg_map[:h - 1, :w - 1] != seg_map[1:h, 1:w])
                    & (seg_map[:h - 1, :w - 1] != self.ignore_index) &
                    (seg_map[1:h, 1:w] != self.ignore_index)] = 1
        # up_right
        edge_upright = edge[:h - 1, 1:w]
        edge_upright[(seg_map[:h - 1, 1:w] != seg_map[1:h, :w - 1])
                     & (seg_map[:h - 1, 1:w] != self.ignore_index) &
                     (seg_map[1:h, :w - 1] != self.ignore_index)] = 1

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (self.edge_width, self.edge_width))
        edge = cv2.dilate(edge, kernel)

        results['gt_edge_map'] = edge
        results['edge_width'] = self.edge_width

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'edge_width={self.edge_width}, '
        repr_str += f'ignore_index={self.ignore_index})'
        return repr_str


@TRANSFORMS.register_module()
class ResizeShortestEdge(BaseTransform):
    """Resize the image and mask while keeping the aspect ratio unchanged.

    Modified from https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/transforms/augmentation_impl.py#L130 # noqa:E501
    Copyright (c) Facebook, Inc. and its affiliates.
    Licensed under the Apache-2.0 License

    This transform attempts to scale the shorter edge to the given
    `scale`, as long as the longer edge does not exceed `max_size`.
    If `max_size` is reached, then downscale so that the longer
    edge does not exceed `max_size`.

    Required Keys:

    - img
    - gt_seg_map (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_seg_map (optional))

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio


    Args:
        scale (Union[int, Tuple[int, int]]): The target short edge length.
            If it's tuple, will select the min value as the short edge length.
        max_size (int): The maximum allowed longest edge length.
    """

    def __init__(self, scale: Union[int, Tuple[int, int]],
                 max_size: int) -> None:
        super().__init__()
        self.scale = scale
        self.max_size = max_size

        # Create a empty Resize object
        self.resize = TRANSFORMS.build({
            'type': 'Resize',
            'scale': 0,
            'keep_ratio': True
        })

    def _get_output_shape(self, img, short_edge_length) -> Tuple[int, int]:
        """Compute the target image shape with the given `short_edge_length`.

        Args:
            img (np.ndarray): The input image.
            short_edge_length (Union[int, Tuple[int, int]]): The target short
                edge length. If it's tuple, will select the min value as the
                short edge length.
        """
        h, w = img.shape[:2]
        if isinstance(short_edge_length, int):
            size = short_edge_length * 1.0
        elif isinstance(short_edge_length, tuple):
            size = min(short_edge_length) * 1.0
        scale = size / min(h, w)
        if h < w:
            new_h, new_w = size, scale * w
        else:
            new_h, new_w = scale * h, size

        if max(new_h, new_w) > self.max_size:
            scale = self.max_size * 1.0 / max(new_h, new_w)
            new_h *= scale
            new_w *= scale

        new_h = int(new_h + 0.5)
        new_w = int(new_w + 0.5)
        return (new_w, new_h)

    def transform(self, results: Dict) -> Dict:
        self.resize.scale = self._get_output_shape(results['img'], self.scale)
        return self.resize(results)


@TRANSFORMS.register_module()
class BioMedical3DRandomCrop(BaseTransform):
    """Crop the input patch for medical image & segmentation mask.

    Required Keys:

    - img (np.ndarray): Biomedical image with shape (N, Z, Y, X),
        N is the number of modalities, and data type is float32.
    - gt_seg_map (np.ndarray, optional): Biomedical semantic segmentation mask
        with shape (Z, Y, X).

    Modified Keys:

        - img
        - img_shape
        - gt_seg_map (optional)

    Args:
        crop_shape (Union[int, Tuple[int, int, int]]):  Expected size after
            cropping with the format of (z, y, x). If set to an integer,
            then cropping width and height are equal to this integer.
        keep_foreground (bool): If keep_foreground is True, it will sample a
            voxel of foreground classes randomly, and will take it as the
            center of the crop bounding-box. Default to True.
    """

    def __init__(self,
                 crop_shape: Union[int, Tuple[int, int, int]],
                 keep_foreground: bool = True):
        super().__init__()
        assert isinstance(crop_shape, int) or (
            isinstance(crop_shape, tuple) and len(crop_shape) == 3
        ), 'The expected crop_shape is an integer, or a tuple containing '
        'three integers'

        if isinstance(crop_shape, int):
            crop_shape = (crop_shape, crop_shape, crop_shape)
        assert crop_shape[0] > 0 and crop_shape[1] > 0 and crop_shape[2] > 0
        self.crop_shape = crop_shape
        self.keep_foreground = keep_foreground

    def random_sample_location(self, seg_map: np.ndarray) -> dict:
        """sample foreground voxel when keep_foreground is True.

        Args:
            seg_map (np.ndarray): gt seg map.

        Returns:
            dict: Coordinates of selected foreground voxel.
        """
        num_samples = 10000
        # at least 1% of the class voxels need to be selected,
        # otherwise it may be too sparse
        min_percent_coverage = 0.01
        class_locs = {}
        foreground_classes = []
        all_classes = np.unique(seg_map)
        for c in all_classes:
            if c == 0:
                # to avoid the segmentation mask full of background 0
                # and the class_locs is just void dictionary {} when it return
                # there add a void list for background 0.
                class_locs[c] = []
            else:
                all_locs = np.argwhere(seg_map == c)
                target_num_samples = min(num_samples, len(all_locs))
                target_num_samples = max(
                    target_num_samples,
                    int(np.ceil(len(all_locs) * min_percent_coverage)))

                selected = all_locs[np.random.choice(
                    len(all_locs), target_num_samples, replace=False)]
                class_locs[c] = selected
                foreground_classes.append(c)

        selected_voxel = None
        if len(foreground_classes) > 0:
            selected_class = np.random.choice(foreground_classes)
            voxels_of_that_class = class_locs[selected_class]
            selected_voxel = voxels_of_that_class[np.random.choice(
                len(voxels_of_that_class))]

        return selected_voxel

    def random_generate_crop_bbox(self, margin_z: int, margin_y: int,
                                  margin_x: int) -> tuple:
        """Randomly get a crop bounding box.

        Args:
            seg_map (np.ndarray): Ground truth segmentation map.

        Returns:
            tuple: Coordinates of the cropped image.
        """
        offset_z = np.random.randint(0, margin_z + 1)
        offset_y = np.random.randint(0, margin_y + 1)
        offset_x = np.random.randint(0, margin_x + 1)
        crop_z1, crop_z2 = offset_z, offset_z + self.crop_shape[0]
        crop_y1, crop_y2 = offset_y, offset_y + self.crop_shape[1]
        crop_x1, crop_x2 = offset_x, offset_x + self.crop_shape[2]

        return crop_z1, crop_z2, crop_y1, crop_y2, crop_x1, crop_x2

    def generate_margin(self, results: dict) -> tuple:
        """Generate margin of crop bounding-box.

        If keep_foreground is True, it will sample a voxel of foreground
        classes randomly, and will take it as the center of the bounding-box,
        and return the margin between of the bounding-box and image.
        If keep_foreground is False, it will return the difference from crop
        shape and image shape.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            tuple: The margin for 3 dimensions of crop bounding-box and image.
        """

        seg_map = results['gt_seg_map']
        if self.keep_foreground:
            selected_voxel = self.random_sample_location(seg_map)
            if selected_voxel is None:
                # this only happens if some image does not contain
                # foreground voxels at all
                warnings.warn(f'case does not contain any foreground classes'
                              f': {results["img_path"]}')
                margin_z = max(seg_map.shape[0] - self.crop_shape[0], 0)
                margin_y = max(seg_map.shape[1] - self.crop_shape[1], 0)
                margin_x = max(seg_map.shape[2] - self.crop_shape[2], 0)
            else:
                margin_z = max(0, selected_voxel[0] - self.crop_shape[0] // 2)
                margin_y = max(0, selected_voxel[1] - self.crop_shape[1] // 2)
                margin_x = max(0, selected_voxel[2] - self.crop_shape[2] // 2)
                margin_z = max(
                    0, min(seg_map.shape[0] - self.crop_shape[0], margin_z))
                margin_y = max(
                    0, min(seg_map.shape[1] - self.crop_shape[1], margin_y))
                margin_x = max(
                    0, min(seg_map.shape[2] - self.crop_shape[2], margin_x))
        else:
            margin_z = max(seg_map.shape[0] - self.crop_shape[0], 0)
            margin_y = max(seg_map.shape[1] - self.crop_shape[1], 0)
            margin_x = max(seg_map.shape[2] - self.crop_shape[2], 0)

        return margin_z, margin_y, margin_x

    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop from ``img``

        Args:
            img (np.ndarray): Original input image.
            crop_bbox (tuple): Coordinates of the cropped image.

        Returns:
            np.ndarray: The cropped image.
        """
        crop_z1, crop_z2, crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        if len(img.shape) == 3:
            # crop seg map
            img = img[crop_z1:crop_z2, crop_y1:crop_y2, crop_x1:crop_x2]
        else:
            # crop image
            assert len(img.shape) == 4
            img = img[:, crop_z1:crop_z2, crop_y1:crop_y2, crop_x1:crop_x2]
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        margin = self.generate_margin(results)
        crop_bbox = self.random_generate_crop_bbox(*margin)

        # crop the image
        img = results['img']
        results['img'] = self.crop(img, crop_bbox)
        results['img_shape'] = results['img'].shape[1:]

        # crop semantic seg
        seg_map = results['gt_seg_map']
        results['gt_seg_map'] = self.crop(seg_map, crop_bbox)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_shape={self.crop_shape})'


@TRANSFORMS.register_module()
class BioMedicalGaussianNoise(BaseTransform):
    """Add random Gaussian noise to image.

    Modified from https://github.com/MIC-DKFZ/batchgenerators/blob/7651ece69faf55263dd582a9f5cbd149ed9c3ad0/batchgenerators/transforms/noise_transforms.py#L53  # noqa:E501

    Copyright (c) German Cancer Research Center (DKFZ)
    Licensed under the Apache License, Version 2.0

    Required Keys:

    - img (np.ndarray): Biomedical image with shape (N, Z, Y, X),
            N is the number of modalities, and data type is float32.

    Modified Keys:

    - img

    Args:
        prob (float): Probability to add Gaussian noise for
            each sample. Default to 0.1.
        mean (float): Mean or “centre” of the distribution. Default to 0.0.
        std (float): Standard deviation of distribution. Default to 0.1.
    """

    def __init__(self,
                 prob: float = 0.1,
                 mean: float = 0.0,
                 std: float = 0.1) -> None:
        super().__init__()
        assert 0.0 <= prob <= 1.0 and std >= 0.0
        self.prob = prob
        self.mean = mean
        self.std = std

    def transform(self, results: Dict) -> Dict:
        """Call function to add random Gaussian noise to image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with random Gaussian noise.
        """
        if np.random.rand() < self.prob:
            rand_std = np.random.uniform(0, self.std)
            noise = np.random.normal(
                self.mean, rand_std, size=results['img'].shape)
            # noise is float64 array, convert to the results['img'].dtype
            noise = noise.astype(results['img'].dtype)
            results['img'] = results['img'] + noise
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'mean={self.mean}, '
        repr_str += f'std={self.std})'
        return repr_str


@TRANSFORMS.register_module()
class BioMedicalGaussianBlur(BaseTransform):
    """Add Gaussian blur with random sigma to image.

    Modified from https://github.com/MIC-DKFZ/batchgenerators/blob/7651ece69faf55263dd582a9f5cbd149ed9c3ad0/batchgenerators/transforms/noise_transforms.py#L81 # noqa:E501

    Copyright (c) German Cancer Research Center (DKFZ)
    Licensed under the Apache License, Version 2.0

    Required Keys:

    - img (np.ndarray): Biomedical image with shape (N, Z, Y, X),
            N is the number of modalities, and data type is float32.

    Modified Keys:

    - img

    Args:
        sigma_range (Tuple[float, float]|float): range to randomly
            select sigma value. Default to (0.5, 1.0).
        prob (float): Probability to apply Gaussian blur
            for each sample. Default to 0.2.
        prob_per_channel  (float): Probability to apply Gaussian blur
            for each channel (axis N of the image). Default to 0.5.
        different_sigma_per_channel (bool): whether to use different
            sigma for each channel (axis N of the image). Default to True.
        different_sigma_per_axis (bool): whether to use different
            sigma for axis Z, X and Y of the image. Default to True.
    """

    def __init__(self,
                 sigma_range: Tuple[float, float] = (0.5, 1.0),
                 prob: float = 0.2,
                 prob_per_channel: float = 0.5,
                 different_sigma_per_channel: bool = True,
                 different_sigma_per_axis: bool = True) -> None:
        super().__init__()
        assert 0.0 <= prob <= 1.0
        assert 0.0 <= prob_per_channel <= 1.0
        assert isinstance(sigma_range, Sequence) and len(sigma_range) == 2
        self.sigma_range = sigma_range
        self.prob = prob
        self.prob_per_channel = prob_per_channel
        self.different_sigma_per_channel = different_sigma_per_channel
        self.different_sigma_per_axis = different_sigma_per_axis

    def _get_valid_sigma(self, value_range) -> Tuple[float, ...]:
        """Ensure the `value_range` to be either a single value or a sequence
        of two values. If the `value_range` is a sequence, generate a random
        value with `[value_range[0], value_range[1]]` based on uniform
        sampling.

        Modified from https://github.com/MIC-DKFZ/batchgenerators/blob/7651ece69faf55263dd582a9f5cbd149ed9c3ad0/batchgenerators/augmentations/utils.py#L625 # noqa:E501

        Args:
            value_range (tuple|list|float|int): the input value range
        """
        if (isinstance(value_range, (list, tuple))):
            if (value_range[0] == value_range[1]):
                value = value_range[0]
            else:
                orig_type = type(value_range[0])
                value = np.random.uniform(value_range[0], value_range[1])
                value = orig_type(value)
        return value

    def _gaussian_blur(self, data_sample: np.ndarray) -> np.ndarray:
        """Random generate sigma and apply Gaussian Blur to the data
        Args:
            data_sample (np.ndarray): data sample with multiple modalities,
                the data shape is (N, Z, Y, X)
        """
        sigma = None
        for c in range(data_sample.shape[0]):
            if np.random.rand() < self.prob_per_channel:
                # if no `sigma` is generated, generate one
                # if `self.different_sigma_per_channel` is True,
                # re-generate random sigma for each channel
                if (sigma is None or self.different_sigma_per_channel):
                    if (not self.different_sigma_per_axis):
                        sigma = self._get_valid_sigma(self.sigma_range)
                    else:
                        sigma = [
                            self._get_valid_sigma(self.sigma_range)
                            for _ in data_sample.shape[1:]
                        ]
                # apply gaussian filter with `sigma`
                data_sample[c] = gaussian_filter(
                    data_sample[c], sigma, order=0)
        return data_sample

    def transform(self, results: Dict) -> Dict:
        """Call function to add random Gaussian blur to image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with random Gaussian noise.
        """
        if np.random.rand() < self.prob:
            results['img'] = self._gaussian_blur(results['img'])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'prob_per_channel={self.prob_per_channel}, '
        repr_str += f'sigma_range={self.sigma_range}, '
        repr_str += 'different_sigma_per_channel='\
                    f'{self.different_sigma_per_channel}, '
        repr_str += 'different_sigma_per_axis='\
                    f'{self.different_sigma_per_axis})'
        return repr_str


@TRANSFORMS.register_module()
class BioMedicalRandomGamma(BaseTransform):
    """Using random gamma correction to process the biomedical image.

    Modified from
    https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/transforms/color_transforms.py#L132 # noqa:E501
    With licence: Apache 2.0

    Required Keys:

    - img (np.ndarray): Biomedical image with shape (N, Z, Y, X),
        N is the number of modalities, and data type is float32.

    Modified Keys:
    - img

    Args:
        prob (float): The probability to perform this transform. Default: 0.5.
        gamma_range (Tuple[float]): Range of gamma values. Default: (0.5, 2).
        invert_image (bool): Whether invert the image before applying gamma
            augmentation. Default: False.
        per_channel (bool): Whether perform the transform each channel
            individually. Default: False
        retain_stats (bool): Gamma transformation will alter the mean and std
            of the data in the patch. If retain_stats=True, the data will be
            transformed to match the mean and standard deviation before gamma
            augmentation. Default: False.
    """

    def __init__(self,
                 prob: float = 0.5,
                 gamma_range: Tuple[float] = (0.5, 2),
                 invert_image: bool = False,
                 per_channel: bool = False,
                 retain_stats: bool = False):
        assert 0 <= prob and prob <= 1
        assert isinstance(gamma_range, tuple) and len(gamma_range) == 2
        assert isinstance(invert_image, bool)
        assert isinstance(per_channel, bool)
        assert isinstance(retain_stats, bool)
        self.prob = prob
        self.gamma_range = gamma_range
        self.invert_image = invert_image
        self.per_channel = per_channel
        self.retain_stats = retain_stats

    @cache_randomness
    def _do_gamma(self):
        """Whether do adjust gamma for image."""
        return np.random.rand() < self.prob

    def _adjust_gamma(self, img: np.array):
        """Gamma adjustment for image.

        Args:
            img (np.array): Input image before gamma adjust.

        Returns:
            np.arrays: Image after gamma adjust.
        """

        if self.invert_image:
            img = -img

        def _do_adjust(img):
            if retain_stats_here:
                img_mean = img.mean()
                img_std = img.std()
            if np.random.random() < 0.5 and self.gamma_range[0] < 1:
                gamma = np.random.uniform(self.gamma_range[0], 1)
            else:
                gamma = np.random.uniform(
                    max(self.gamma_range[0], 1), self.gamma_range[1])
            img_min = img.min()
            img_range = img.max() - img_min  # range
            img = np.power(((img - img_min) / float(img_range + 1e-7)),
                           gamma) * img_range + img_min
            if retain_stats_here:
                img = img - img.mean()
                img = img / (img.std() + 1e-8) * img_std
                img = img + img_mean
            return img

        if not self.per_channel:
            retain_stats_here = self.retain_stats
            img = _do_adjust(img)
        else:
            for c in range(img.shape[0]):
                img[c] = _do_adjust(img[c])
        if self.invert_image:
            img = -img
        return img

    def transform(self, results: dict) -> dict:
        """Call function to perform random gamma correction
        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with random gamma correction performed.
        """
        do_gamma = self._do_gamma()

        if do_gamma:
            results['img'] = self._adjust_gamma(results['img'])
        else:
            pass
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'gamma_range={self.gamma_range},'
        repr_str += f'invert_image={self.invert_image},'
        repr_str += f'per_channel={self.per_channel},'
        repr_str += f'retain_stats={self.retain_stats}'
        return repr_str


@TRANSFORMS.register_module()
class BioMedical3DPad(BaseTransform):
    """Pad the biomedical 3d image & biomedical 3d semantic segmentation maps.

    Required Keys:

    - img (np.ndarry): Biomedical image with shape (N, Z, Y, X) by default,
        N is the number of modalities.
    - gt_seg_map (np.ndarray, optional): Biomedical seg map with shape
        (Z, Y, X) by default.

    Modified Keys:

    - img (np.ndarry): Biomedical image with shape (N, Z, Y, X) by default,
        N is the number of modalities.
    - gt_seg_map (np.ndarray, optional): Biomedical seg map with shape
        (Z, Y, X) by default.

    Added Keys:

    - pad_shape (Tuple[int, int, int]): The padded shape.

    Args:
        pad_shape (Tuple[int, int, int]): Fixed padding size.
            Expected padding shape (Z, Y, X).
        pad_val (float): Padding value for biomedical image.
            The padding mode is set to "constant". The value
            to be filled in padding area. Default: 0.
        seg_pad_val (int): Padding value for biomedical 3d semantic
            segmentation maps. The padding mode is set to "constant".
            The value to be filled in padding area. Default: 0.
    """

    def __init__(self,
                 pad_shape: Tuple[int, int, int],
                 pad_val: float = 0.,
                 seg_pad_val: int = 0) -> None:

        # check pad_shape
        assert pad_shape is not None
        if not isinstance(pad_shape, tuple):
            assert len(pad_shape) == 3

        self.pad_shape = pad_shape
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    def _pad_img(self, results: dict) -> None:
        """Pad images according to ``self.pad_shape``

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: The dict contains the padded image and shape
                information.
        """
        padded_img = self._to_pad(
            results['img'], pad_shape=self.pad_shape, pad_val=self.pad_val)

        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape[1:]

    def _pad_seg(self, results: dict) -> None:
        """Pad semantic segmentation map according to ``self.pad_shape`` if
        ``gt_seg_map`` is not None in results dict.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Update the padded gt seg map in dict.
        """
        if results.get('gt_seg_map', None) is not None:
            pad_gt_seg = self._to_pad(
                results['gt_seg_map'][None, ...],
                pad_shape=results['pad_shape'],
                pad_val=self.seg_pad_val)
            results['gt_seg_map'] = pad_gt_seg[1:]

    @staticmethod
    def _to_pad(img: np.ndarray,
                pad_shape: Tuple[int, int, int],
                pad_val: Union[int, float] = 0) -> np.ndarray:
        """Pad the given 3d image to a certain shape with specified padding
        value.

        Args:
            img (ndarray): Biomedical image with shape (N, Z, Y, X)
                to be padded. N is the number of modalities.
            pad_shape (Tuple[int,int,int]): Expected padding shape (Z, Y, X).
            pad_val (float, int): Values to be filled in padding areas
                and the padding_mode is set to 'constant'. Default: 0.

        Returns:
            ndarray: The padded image.
        """
        # compute pad width
        d = max(pad_shape[0] - img.shape[1], 0)
        pad_d = (d // 2, d - d // 2)
        h = max(pad_shape[1] - img.shape[2], 0)
        pad_h = (h // 2, h - h // 2)
        w = max(pad_shape[2] - img.shape[2], 0)
        pad_w = (w // 2, w - w // 2)

        pad_list = [(0, 0), pad_d, pad_h, pad_w]

        img = np.pad(img, pad_list, mode='constant', constant_values=pad_val)
        return img

    def transform(self, results: dict) -> dict:
        """Call function to pad images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_seg(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'pad_shape={self.pad_shape}, '
        repr_str += f'pad_val={self.pad_val}), '
        repr_str += f'seg_pad_val={self.seg_pad_val})'
        return repr_str


@TRANSFORMS.register_module()
class BioMedical3DRandomFlip(BaseTransform):
    """Flip biomedical 3D images and segmentations.

    Modified from https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/transforms/spatial_transforms.py # noqa:E501

    Copyright 2021 Division of
    Medical Image Computing, German Cancer Research Center (DKFZ) and Applied
    Computer Vision Lab, Helmholtz Imaging Platform.
    Licensed under the Apache-2.0 License.

    Required Keys:

    - img (np.ndarry): Biomedical image with shape (N, Z, Y, X) by default,
        N is the number of modalities.
    - gt_seg_map (np.ndarray, optional): Biomedical seg map with shape
        (Z, Y, X) by default.

    Modified Keys:

    - img (np.ndarry): Biomedical image with shape (N, Z, Y, X) by default,
        N is the number of modalities.
    - gt_seg_map (np.ndarray, optional): Biomedical seg map with shape
        (Z, Y, X) by default.

    Added Keys:

    - do_flip
    - flip_axes

    Args:
        prob (float): Flipping probability.
        axes (Tuple[int, ...]): Flipping axes with order 'ZXY'.
        swap_label_pairs (Optional[List[Tuple[int, int]]]):
        The segmentation label pairs that are swapped when flipping.
    """

    def __init__(self,
                 prob: float,
                 axes: Tuple[int, ...],
                 swap_label_pairs: Optional[List[Tuple[int, int]]] = None):
        self.prob = prob
        self.axes = axes
        self.swap_label_pairs = swap_label_pairs
        assert prob >= 0 and prob <= 1
        if axes is not None:
            assert max(axes) <= 2

    @staticmethod
    def _flip(img, direction: Tuple[bool, bool, bool]) -> np.ndarray:
        if direction[0]:
            img[:, :] = img[:, ::-1]
        if direction[1]:
            img[:, :, :] = img[:, :, ::-1]
        if direction[2]:
            img[:, :, :, :] = img[:, :, :, ::-1]
        return img

    def _do_flip(self, img: np.ndarray) -> Tuple[bool, bool, bool]:
        """Call function to determine which axis to flip.

        Args:
            img (np.ndarry): Image or segmentation map array.
        Returns:
            tuple: Flip action, whether to flip on the z, x, and y axes.
        """
        flip_c, flip_x, flip_y = False, False, False
        if self.axes is not None:
            flip_c = 0 in self.axes and np.random.rand() < self.prob
            flip_x = 1 in self.axes and np.random.rand() < self.prob
            if len(img.shape) == 4:
                flip_y = 2 in self.axes and np.random.rand() < self.prob
        return flip_c, flip_x, flip_y

    def _swap_label(self, seg: np.ndarray) -> np.ndarray:
        out = seg.copy()
        for first, second in self.swap_label_pairs:
            first_area = (seg == first)
            second_area = (seg == second)
            out[first_area] = second
            out[second_area] = first
        return out

    def transform(self, results: Dict) -> Dict:
        """Call function to flip and swap pair labels.

        Args:
            results (dict): Result dict.
        Returns:
            dict: Flipped results, 'do_flip', 'flip_axes' keys are added into
                result dict.
        """
        # get actual flipped axis
        if 'do_flip' not in results:
            results['do_flip'] = self._do_flip(results['img'])
        if 'flip_axes' not in results:
            results['flip_axes'] = self.axes
        # flip image
        results['img'] = self._flip(
            results['img'], direction=results['do_flip'])
        # flip seg
        if results['gt_seg_map'] is not None:
            if results['gt_seg_map'].shape != results['img'].shape:
                results['gt_seg_map'] = results['gt_seg_map'][None, :]
            results['gt_seg_map'] = self._flip(
                results['gt_seg_map'], direction=results['do_flip'])
            results['gt_seg_map'] = results['gt_seg_map'].squeeze()
            # swap label pairs
            if self.swap_label_pairs is not None:
                results['gt_seg_map'] = self._swap_label(results['gt_seg_map'])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, axes={self.axes}, ' \
                    f'swap_label_pairs={self.swap_label_pairs})'
        return repr_str


''' Customized '''
@TRANSFORMS.register_module()
class RandomCropWithExtra(BaseTransform):
    """Random crop the image & seg.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - img_shape
    - gt_seg_map


    Args:
        crop_size (Union[int, Tuple[int, int]]):  Expected size after cropping
            with the format of (h, w). If set to an integer, then cropping
            width and height are equal to this integer.
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
        ignore_index (int): The label index to be ignored. Default: 255
    """

    def __init__(self,
                 crop_size: Union[int, Tuple[int, int]],
                 extra_keys=(),
                 stride=None,
                 cat_max_ratio: float = 1.,
                 ignore_index: int = 255):
        super().__init__()
        assert isinstance(crop_size, int) or (
            isinstance(crop_size, tuple) and len(crop_size) == 2
        ), 'The expected crop_size is an integer, or a tuple containing two '
        'intergers'

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        if stride is not None:
            assert crop_size[0] % stride == 0 and crop_size[1] % stride == 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index
        self.stride = stride
        self.extra_keys = extra_keys
        if isinstance(self.extra_keys, str):
            self.extra_keys = (self.extra_keys, )

    @cache_randomness
    def crop_bbox(self, results: dict) -> tuple:
        """get a crop bounding box.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            tuple: Coordinates of the cropped image.
        """

        def generate_crop_bbox(img: np.ndarray) -> tuple:
            """Randomly get a crop bounding box.

            Args:
                img (np.ndarray): Original input image.

            Returns:
                tuple: Coordinates of the cropped image.
            """

            margin_h = max(img.shape[0] - self.crop_size[0], 0)
            margin_w = max(img.shape[1] - self.crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)

            if self.stride is not None:
                offset_h = int(offset_h / self.stride) * self.stride
                offset_w = int(offset_w / self.stride) * self.stride

            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            return crop_y1, crop_y2, crop_x1, crop_x2

        img = results['img']
        crop_bbox = generate_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_seg_map'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = generate_crop_bbox(img)

        return crop_bbox

    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop from ``img``

        Args:
            img (np.ndarray): Original input image.
            crop_bbox (tuple): Coordinates of the cropped image.

        Returns:
            np.ndarray: The cropped image.
        """

        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        crop_bbox = self.crop_bbox(results)

        # crop the image
        img = self.crop(img, crop_bbox)

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        # crop extra fields
        for key in self.extra_keys:
            if results.get(key, None) is not None:
                extra_info = results[key]
                extra_info = self.crop(extra_info, crop_bbox)
                results[key] = extra_info


        results['img'] = img
        results['img_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size},' \
                    f'extra_keys={self.extra_keys}, ' \
                    f'stride={self.stride}, ' \
                    f'ignore_index={self.ignore_index})'
        return repr_str



@TRANSFORMS.register_module()
class RandomCropWithDCT(BaseTransform):
    """Random crop the image & seg & block DCT.
    The crop stride is multiple to 8

    Required Keys:

    - img
    - dct
    - gt_seg_map

    Modified Keys:

    - img
    - dct
    - img_shape
    - gt_seg_map


    Args:
        crop_size (Union[int, Tuple[int, int]]):  Expected size after cropping
            with the format of (h, w). If set to an integer, then cropping
            width and height are equal to this integer.
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
        ignore_index (int): The label index to be ignored. Default: 255
    """

    def __init__(self,
                 crop_size: Union[int, Tuple[int, int]],
                 cat_max_ratio: float = 1.,
                 target_cat = None,
                 cat_specified_ratio = 0.5,
                 ignore_index: int = 255):
        super().__init__()
        assert isinstance(crop_size, int) or (
                isinstance(crop_size, tuple) and len(crop_size) == 2
        ), 'The expected crop_size is an integer, or a tuple containing two '
        'intergers'

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        assert crop_size[0] % 8 == 0 and crop_size[1] % 8 == 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index
        self.target_cat = target_cat
        self.cat_specified_ratio = cat_specified_ratio

    @cache_randomness
    def crop_bbox(self, results: dict) -> tuple:
        """get a crop bounding box.
        The crop stride is multiple to 8

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            tuple: Coordinates of the cropped image.
        """

        def generate_crop_bbox(img: np.ndarray) -> tuple:
            """Randomly get a crop bounding box.

            Args:
                img (np.ndarray): Original input image.

            Returns:
                tuple: Coordinates of the cropped image.
            """

            margin_h = max(img.shape[0] - self.crop_size[0], 0)
            margin_w = max(img.shape[1] - self.crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)

            offset_h = offset_h // 8 * 8
            offset_w = offset_w // 8 * 8

            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            return crop_y1, crop_y2, crop_x1, crop_x2

        img = results['img']
        crop_bbox = generate_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_seg_map'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]

                if self.target_cat is not None:
                    if len(cnt) > 1 and np.max(cnt) / np.sum(
                            cnt) < self.cat_max_ratio and cnt[self.target_cat] / np.sum(
                            cnt) < self.cat_specified_ratio:
                        break
                else:
                    if len(cnt) > 1 and np.max(cnt) / np.sum(
                            cnt) < self.cat_max_ratio:
                        break
                crop_bbox = generate_crop_bbox(img)

        return crop_bbox

    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop from ``img``

        Args:
            img (np.ndarray): Original input image.
            crop_bbox (tuple): Coordinates of the cropped image.

        Returns:
            np.ndarray: The cropped image.
        """

        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        crop_bbox = self.crop_bbox(results)

        # crop the image
        img = self.crop(img, crop_bbox)

        # crop the dct
        if results.get('dct', None) is not None:
            dct = results['dct']
            dct = self.crop(dct, crop_bbox)
            results['dct'] = dct

        # crop the ocr map
        if results.get('ocr', None) is not None:
            ocr = results['ocr']
            ocr = self.crop(ocr, crop_bbox)
            results['ocr'] = ocr

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'

# TODO:
@TRANSFORMS.register_module()
class ProcessDCT(BaseTransform):
    """Process DCT coef, modified from CAT-Net"""

    def __init__(self, to_vol=True, threshold=0):
        self.to_vol = to_vol
        self.threshold = threshold
        if threshold != 0:
            assert threshold >= 1

    def transform(self, results: dict) -> dict:
        assert 'dct' in results

        results['dct'] = np.abs(results['dct'])
        results['dct'] = np.round(results['dct'])

        if self.to_vol:
            DCT_vol = np.zeros((self.threshold + 1, results['dct'].shape[0], results['dct'].shape[1]))
            DCT_vol[0] += (results['dct'] == 0)

            for i in range(1, self.threshold):
                DCT_vol[i] += (results['dct'] == i)
            DCT_vol[self.threshold] += (results['dct'] >= self.threshold)

            results['dct'] = DCT_vol
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'

@TRANSFORMS.register_module()
class ResizeShortestEdgeWithDCT(BaseTransform):
    """Resize the image and mask while keeping the aspect ratio unchanged.

    Modified from https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/transforms/augmentation_impl.py#L130 # noqa:E501
    Copyright (c) Facebook, Inc. and its affiliates.
    Licensed under the Apache-2.0 License

    This transform attempts to scale the shorter edge to the given
    `scale`, as long as the longer edge does not exceed `max_size`.
    If `max_size` is reached, then downscale so that the longer
    edge does not exceed `max_size`.

    Required Keys:

    - img
    - dct
    - gt_seg_map (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_seg_map (optional))

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio


    Args:
        scale (Union[int, Tuple[int, int]]): The target short edge length.
            If it's tuple, will select the min value as the short edge length.
        max_size (int): The maximum allowed longest edge length.
    """

    def __init__(self, scale: Union[int, Tuple[int, int]],
                 max_size: int) -> None:
        super().__init__()
        self.scale = scale
        self.max_size = max_size

        # Create a empty Resize object
        self.resize = TRANSFORMS.build({
            'type': 'ResizeWithDCT',
            'scale': 0,
            'keep_ratio': True
        })

    def _get_output_shape(self, img, short_edge_length) -> Tuple[int, int]:
        """Compute the target image shape with the given `short_edge_length`.

        Args:
            img (np.ndarray): The input image.
            short_edge_length (Union[int, Tuple[int, int]]): The target short
                edge length. If it's tuple, will select the min value as the
                short edge length.
        """
        h, w = img.shape[:2]
        if isinstance(short_edge_length, int):
            size = short_edge_length * 1.0
        elif isinstance(short_edge_length, tuple):
            size = min(short_edge_length) * 1.0
        scale = size / min(h, w)
        if h < w:
            new_h, new_w = size, scale * w
        else:
            new_h, new_w = scale * h, size

        if max(new_h, new_w) > self.max_size:
            scale = self.max_size * 1.0 / max(new_h, new_w)
            new_h *= scale
            new_w *= scale

        new_h = int(new_h + 0.5) // 8 * 8
        new_w = int(new_w + 0.5) // 8 * 8
        return (new_w, new_h)

    def transform(self, results: Dict) -> Dict:
        self.resize.scale = self._get_output_shape(results['img'], self.scale)
        return self.resize(results)

@TRANSFORMS.register_module()
class ResizeWithDCT(BaseTransform):
    """Resize images & bbox & seg & keypoints.

    This transform resizes the input image according to ``scale`` or
    ``scale_factor``. Bboxes, seg map and keypoints are then resized with the
    same scale factor.
    if ``scale`` and ``scale_factor`` are both set, it will use ``scale`` to
    resize.

    Required Keys:

    - img
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_keypoints (optional)

    Modified Keys:

    - img
    - gt_bboxes
    - gt_seg_map
    - gt_keypoints
    - img_shape

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio

    Args:
        scale (int or tuple): Images scales for resizing. Defaults to None
        scale_factor (float or tuple[float]): Scale factors for resizing.
            Defaults to None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Defaults to False.
        clip_object_border (bool): Whether to clip the objects
            outside the border of the image. In some dataset like MOT17, the gt
            bboxes are allowed to cross the border of images. Therefore, we
            don't need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 scale: Optional[Union[int, Tuple[int, int]]] = None,
                 scale_factor: Optional[Union[float, Tuple[float,
                                                           float]]] = None,
                 keep_ratio: bool = False,
                 clip_object_border: bool = True,
                 backend: str = 'cv2',
                 interpolation='bilinear') -> None:
        assert scale is not None or scale_factor is not None, (
            '`scale` and'
            '`scale_factor` can not both be `None`')
        if scale is None:
            self.scale = None
        else:
            if isinstance(scale, int):
                self.scale = (scale, scale)
            else:
                self.scale = scale

        self.backend = backend
        self.interpolation = interpolation
        self.keep_ratio = keep_ratio
        self.clip_object_border = clip_object_border
        if scale_factor is None:
            self.scale_factor = None
        elif isinstance(scale_factor, float):
            self.scale_factor = (scale_factor, scale_factor)
        elif isinstance(scale_factor, tuple):
            assert (len(scale_factor)) == 2
            self.scale_factor = scale_factor
        else:
            raise TypeError(
                f'expect scale_factor is float or Tuple(float), but'
                f'get {type(scale_factor)}')

    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""

        if results.get('img', None) is not None:
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results['img'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results['img'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results['img'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['scale_factor'] = (w_scale, h_scale)
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_bboxes', None) is not None:
            bboxes = results['gt_bboxes'] * np.tile(
                np.array(results['scale_factor']), 2)
            if self.clip_object_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0,
                                          results['img_shape'][1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0,
                                          results['img_shape'][0])
            results['gt_bboxes'] = bboxes

    def _resize_seg(self, results: dict) -> None:
        """Resize semantic segmentation map with ``results['scale']``."""
        if results.get('gt_seg_map', None) is not None:
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results['gt_seg_map'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results['gt_seg_map'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_seg_map'] = gt_seg

    def _resize_keypoints(self, results: dict) -> None:
        """Resize keypoints with ``results['scale_factor']``."""
        if results.get('gt_keypoints', None) is not None:
            keypoints = results['gt_keypoints']

            keypoints[:, :, :2] = keypoints[:, :, :2] * np.array(
                results['scale_factor'])
            if self.clip_object_border:
                keypoints[:, :, 0] = np.clip(keypoints[:, :, 0], 0,
                                             results['img_shape'][1])
                keypoints[:, :, 1] = np.clip(keypoints[:, :, 1], 0,
                                             results['img_shape'][0])
            results['gt_keypoints'] = keypoints

    def _resize_dct(self, results: dict) -> None:
        """Resize dct with ``results['scale_factor']``."""
        if results.get('dct', None) is not None:
            block_dct = results['dct']
            dct_h, dct_w = block_dct.shape[0], block_dct.shape[1]
            assert dct_h % 8 == 0
            assert dct_w % 8 == 0
            block_dct = block_dct.reshape(dct_h // 8, 8, dct_w // 8, 8).transpose(0, 2, 1, 3)
            block_dct = block_dct.reshape(dct_h // 8, dct_w // 8, 64)

            block_dct = torch.from_numpy(block_dct.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0)

            w_scale, h_scale = results['scale_factor']
            resize_h = int(np.ceil(dct_h // 8 * h_scale))
            resize_w = int(np.ceil(dct_w // 8 * w_scale))

            dct_map = F.interpolate(block_dct, size=(resize_h, resize_w), mode='bilinear')
            new_h, new_w = dct_map.shape[2] * 8, dct_map.shape[3] * 8
            dct_map = dct_map.squeeze(0).numpy().transpose(1, 2, 0)

            dct_map = dct_map.reshape(new_h // 8, new_w // 8, 8, 8).transpose(0, 2, 1, 3)
            dct_map = dct_map.reshape(new_h, new_w)

            results['dct'] = dct_map

    def _resize_ocr(self, results: dict) -> None:
        """Resize dct with ``results['scale_factor']``."""
        if results.get('ocr', None) is not None:
            if self.keep_ratio:
                ocr_map = mmcv.imrescale(
                    results['ocr'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                ocr_map = mmcv.imresize(
                    results['ocr'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['ocr'] = ocr_map

    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map and keypoints.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'scale', 'scale_factor', 'img_shape',
            and 'keep_ratio' keys are updated in result dict.
        """

        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1],
                                           self.scale_factor)  # type: ignore
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_seg(results)
        self._resize_keypoints(results)
        self._resize_dct(results)
        self._resize_ocr(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'scale_factor={self.scale_factor}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'clip_object_border={self.clip_object_border}), '
        repr_str += f'backend={self.backend}), '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str

@TRANSFORMS.register_module()
class PadWithDCT(BaseTransform):
    """Pad the image & segmentation map.

    There are three padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number. and (3)pad to square. Also,
    pad to square and pad to the minimum size can be used as the same time.

    Required Keys:

    - img
    - gt_bboxes (optional)
    - gt_seg_map (optional)

    Modified Keys:

    - img
    - gt_seg_map
    - img_shape

    Added Keys:

    - pad_shape
    - pad_fixed_size
    - pad_size_divisor

    Args:
        size (tuple, optional): Fixed padding size.
            Expected padding shape (w, h). Defaults to None.
        size_divisor (int, optional): The divisor of padded size. Defaults to
            None.
        pad_to_square (bool): Whether to pad the image into a square.
            Currently only used for YOLOX. Defaults to False.
        pad_val (Number | dict[str, Number], optional): Padding value for if
            the pad_mode is "constant". If it is a single number, the value
            to pad the image is the number and to pad the semantic
            segmentation map is 255. If it is a dict, it should have the
            following keys:

            - img: The value to pad the image.
            - seg: The value to pad the semantic segmentation map.

            Defaults to dict(img=0, seg=255).
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Defaults to 'constant'.

            - constant: pads with a constant value, this value is specified
              with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self,
                 size: Optional[Tuple[int, int]] = None,
                 size_divisor: Optional[int] = None,
                 pad_to_square: bool = False,
                 pad_val: Union[Number, dict] = dict(img=0, seg=255, dct=0, ocr=0),
                 padding_mode: str = 'constant') -> None:
        self.size = size
        self.size_divisor = size_divisor

        if size_divisor is not None:
            assert size_divisor % 8 == 0

        if isinstance(pad_val, int):
            pad_val = dict(img=pad_val, seg=255, dct=0)
        assert isinstance(pad_val, dict), 'pad_val '
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square

        if pad_to_square:
            assert size is None, \
                'The size and size_divisor must be None ' \
                'when pad2square is True'
        else:
            assert size is not None or size_divisor is not None, \
                'only one of size and size_divisor should be valid'
            assert size is None or size_divisor is None
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.padding_mode = padding_mode

    def _pad_img(self, results: dict) -> None:
        """Pad images according to ``self.size``."""
        pad_val = self.pad_val.get('img', 0)

        size = None
        if self.pad_to_square:
            max_size = max(results['img'].shape[:2])
            size = (max_size, max_size)
        if self.size_divisor is not None:
            if size is None:
                size = (results['img'].shape[0], results['img'].shape[1])
            pad_h = int(np.ceil(
                size[0] / self.size_divisor)) * self.size_divisor
            pad_w = int(np.ceil(
                size[1] / self.size_divisor)) * self.size_divisor
            size = (pad_h, pad_w)
        elif self.size is not None:
            size = self.size[::-1]
        if isinstance(pad_val, int) and results['img'].ndim == 3:
            pad_val = tuple(pad_val for _ in range(results['img'].shape[2]))
        padded_img = mmcv.impad(
            results['img'],
            shape=size,
            pad_val=pad_val,
            padding_mode=self.padding_mode)

        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
        results['img_shape'] = padded_img.shape[:2]

    def _pad_seg(self, results: dict) -> None:
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        if results.get('gt_seg_map', None) is not None:
            pad_val = self.pad_val.get('seg', 0)
            if isinstance(pad_val, int) and results['gt_seg_map'].ndim == 3:
                pad_val = tuple(
                    pad_val for _ in range(results['gt_seg_map'].shape[2]))
            results['gt_seg_map'] = mmcv.impad(
                results['gt_seg_map'],
                shape=results['pad_shape'][:2],
                pad_val=pad_val,
                padding_mode=self.padding_mode)

    def _pad_dct(self, results: dict) -> None:
        if results.get('dct', None) is not None:
            pad_val = self.pad_val.get('dct', 0)
            if isinstance(pad_val, int) and results['dct'].ndim == 3:
                pad_val = tuple(
                    pad_val for _ in range(results['dct'].shape[2]))
            results['dct'] = mmcv.impad(
                results['dct'],
                shape=results['dct'][:2],
                pad_val=pad_val,
                padding_mode=self.padding_mode)

    def _pad_ocr(self, results: dict) -> None:
        if results.get('ocr', None) is not None:
            pad_val = self.pad_val.get('ocr', 0)
            if isinstance(pad_val, int) and results['ocr'].ndim == 3:
                pad_val = tuple(
                    pad_val for _ in range(results['ocr'].shape[2]))
            results['ocr'] = mmcv.impad(
                results['ocr'],
                shape=results['ocr'][:2],
                pad_val=pad_val,
                padding_mode=self.padding_mode)

    def transform(self, results: dict) -> dict:
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_seg(results)
        self._pad_dct(results)
        self._pad_ocr(results)
        assert results['img'].shape[0] == results['dct'].shape[0]
        assert results['img'].shape[1] == results['dct'].shape[1]
        print("Before:", results['pad_shape'])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_to_square={self.pad_to_square}, '
        repr_str += f'pad_val={self.pad_val}), '
        repr_str += f'padding_mode={self.padding_mode})'
        return repr_str

@TRANSFORMS.register_module()
class RandomFlipWithDCT(BaseTransform):
    """Flip the image & bbox & keypoints & segmentation map. Added or Updated
    keys: flip, flip_direction, img, gt_bboxes, gt_seg_map, and
    gt_keypoints. There are 3 flip modes:

    - ``prob`` is float, ``direction`` is string: the image will be
      ``direction``ly flipped with probability of ``prob`` .
      E.g., ``prob=0.5``, ``direction='horizontal'``,
      then image will be horizontally flipped with probability of 0.5.

    - ``prob`` is float, ``direction`` is list of string: the image will
      be ``direction[i]``ly flipped with probability of
      ``prob/len(direction)``.
      E.g., ``prob=0.5``, ``direction=['horizontal', 'vertical']``,
      then image will be horizontally flipped with probability of 0.25,
      vertically with probability of 0.25.

    - ``prob`` is list of float, ``direction`` is list of string:
      given ``len(prob) == len(direction)``, the image will
      be ``direction[i]``ly flipped with probability of ``prob[i]``.
      E.g., ``prob=[0.3, 0.5]``, ``direction=['horizontal',
      'vertical']``, then image will be horizontally flipped with
      probability of 0.3, vertically with probability of 0.5.

    Required Keys:

    - img
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_keypoints (optional)

    Modified Keys:

    - img
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_keypoints (optional)

    Added Keys:

    - flip
    - flip_direction
    - swap_seg_labels (optional)

    Args:
        prob (float | list[float], optional): The flipping probability.
            Defaults to None.
        direction(str | list[str]): The flipping direction. Options
            If input is a list, the length must equal ``prob``. Each
            element in ``prob`` indicates the flip probability of
            corresponding direction. Defaults to 'horizontal'.
        swap_seg_labels (list, optional): The label pair need to be swapped
            for ground truth, like 'left arm' and 'right arm' need to be
            swapped after horizontal flipping. For example, ``[(1, 5)]``,
            where 1/5 is the label of the left/right arm. Defaults to None.
    """

    def __init__(self,
                 prob: Optional[Union[float, Iterable[float]]] = None,
                 direction: Union[str, Sequence[Optional[str]]] = 'horizontal',
                 swap_seg_labels: Optional[Sequence] = None) -> None:
        if isinstance(prob, list):
            assert mmengine.is_list_of(prob, float)
            assert 0 <= sum(prob) <= 1
        elif isinstance(prob, float):
            assert 0 <= prob <= 1
        else:
            raise ValueError(f'probs must be float or list of float, but \
                              got `{type(prob)}`.')
        self.prob = prob
        self.swap_seg_labels = swap_seg_labels

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert mmengine.is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError(f'direction must be either str or list of str, \
                               but got `{type(direction)}`.')
        self.direction = direction

        if isinstance(prob, list):
            assert len(prob) == len(self.direction)

    def _flip_bbox(self, bboxes: np.ndarray, img_shape: Tuple[int, int],
                   direction: str) -> np.ndarray:
        """Flip bboxes horizontally.

        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.
        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        h, w = img_shape
        if direction == 'horizontal':
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        elif direction == 'diagonal':
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        else:
            raise ValueError(
                f"Flipping direction must be 'horizontal', 'vertical', \
                  or 'diagonal', but got '{direction}'")
        return flipped

    def _flip_keypoints(
            self,
            keypoints: np.ndarray,
            img_shape: Tuple[int, int],
            direction: str,
    ) -> np.ndarray:
        """Flip keypoints horizontally, vertically or diagonally.

        Args:
            keypoints (numpy.ndarray): Keypoints, shape (..., 2)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.
        Returns:
            numpy.ndarray: Flipped keypoints.
        """

        meta_info = keypoints[..., 2:]
        keypoints = keypoints[..., :2]
        flipped = keypoints.copy()
        h, w = img_shape
        if direction == 'horizontal':
            flipped[..., 0::2] = w - keypoints[..., 0::2]
        elif direction == 'vertical':
            flipped[..., 1::2] = h - keypoints[..., 1::2]
        elif direction == 'diagonal':
            flipped[..., 0::2] = w - keypoints[..., 0::2]
            flipped[..., 1::2] = h - keypoints[..., 1::2]
        else:
            raise ValueError(
                f"Flipping direction must be 'horizontal', 'vertical', \
                  or 'diagonal', but got '{direction}'")
        flipped = np.concatenate([keypoints, meta_info], axis=-1)
        return flipped

    def _flip_dct_map(self, dct_map, direction: str):
        dct_h, dct_w = dct_map.shape[0], dct_map.shape[1]
        dct_map = dct_map.reshape(dct_h // 8, 8, dct_w // 8, 8).transpose(0, 2, 1, 3)
        dct_map = dct_map.reshape(dct_h // 8, dct_w // 8, 64)

        dct_map = mmcv.imflip(
            dct_map, direction=direction)

        dct_map = dct_map.reshape(dct_h // 8, dct_w // 8, 8, 8).transpose(0, 2, 1, 3)
        dct_map = dct_map.reshape(dct_h, dct_w)

        return dct_map

    def _flip_seg_map(self, seg_map: dict, direction: str) -> np.ndarray:
        """Flip segmentation map horizontally, vertically or diagonally.

        Args:
            seg_map (numpy.ndarray): segmentation map, shape (H, W).
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.

        Returns:
            numpy.ndarray: Flipped segmentation map.
        """
        seg_map = mmcv.imflip(seg_map, direction=direction)
        if self.swap_seg_labels is not None:
            # to handle datasets with left/right annotations
            # like 'Left-arm' and 'Right-arm' in LIP dataset
            # Modified from https://github.com/openseg-group/openseg.pytorch/blob/master/lib/datasets/tools/cv2_aug_transforms.py # noqa:E501
            # Licensed under MIT license
            temp = seg_map.copy()
            assert isinstance(self.swap_seg_labels, (tuple, list))
            for pair in self.swap_seg_labels:
                assert isinstance(pair, (tuple, list)) and len(pair) == 2, \
                    'swap_seg_labels must be a sequence with pair, but got ' \
                    f'{self.swap_seg_labels}.'
                seg_map[temp == pair[0]] = pair[1]
                seg_map[temp == pair[1]] = pair[0]
        return seg_map

    @cache_randomness
    def _choose_direction(self) -> str:
        """Choose the flip direction according to `prob` and `direction`"""
        if isinstance(self.direction,
                      Sequence) and not isinstance(self.direction, str):
            # None means non-flip
            direction_list: list = list(self.direction) + [None]
        elif isinstance(self.direction, str):
            # None means non-flip
            direction_list = [self.direction, None]

        if isinstance(self.prob, list):
            non_prob: float = 1 - sum(self.prob)
            prob_list = self.prob + [non_prob]
        elif isinstance(self.prob, float):
            non_prob = 1. - self.prob
            # exclude non-flip
            single_ratio = self.prob / (len(direction_list) - 1)
            prob_list = [single_ratio] * (len(direction_list) - 1) + [non_prob]

        cur_dir = np.random.choice(direction_list, p=prob_list)

        return cur_dir

    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, semantic segmentation map and
        keypoints."""
        # flip image
        results['img'] = mmcv.imflip(
            results['img'], direction=results['flip_direction'])

        img_shape = results['img'].shape[:2]

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'] = self._flip_bbox(results['gt_bboxes'],
                                                   img_shape,
                                                   results['flip_direction'])

        # flip keypoints
        if results.get('gt_keypoints', None) is not None:
            results['gt_keypoints'] = self._flip_keypoints(
                results['gt_keypoints'], img_shape, results['flip_direction'])

        # flip seg map
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = self._flip_seg_map(
                results['gt_seg_map'], direction=results['flip_direction'])
            results['swap_seg_labels'] = self.swap_seg_labels

        # flip dct map
        if results.get('dct', None) is not None:
            results['dct'] = self._flip_dct_map(
                results['dct'], direction=results['flip_direction'])

        # flip ocr map
        if results.get('ocr', None) is not None:
            results['ocr'] = mmcv.imflip(
                results['ocr'], direction=results['flip_direction'])

    def _flip_on_direction(self, results: dict) -> None:
        """Function to flip images, bounding boxes, semantic segmentation map
        and keypoints."""
        cur_dir = self._choose_direction()
        if cur_dir is None:
            results['flip'] = False
            results['flip_direction'] = None
        else:
            results['flip'] = True
            results['flip_direction'] = cur_dir
            self._flip(results)

    def transform(self, results: dict) -> dict:
        """Transform function to flip images, bounding boxes, semantic
        segmentation map and keypoints.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Flipped results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'flip', and 'flip_direction' keys are
            updated in result dict.
        """
        self._flip_on_direction(results)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'direction={self.direction})'

        return repr_str

@TRANSFORMS.register_module()
class OneHot(BaseTransform):
    def __init__(self, keys=('ocr')):
        self.keys = keys

    def _one_hot(self, mask):
        assert len(mask.shape) == 2
        return np.eye(2)[mask]

    def transform(self, results: dict) -> dict:
        for key in self.keys:
            if results.get(key, None) is not None:
                mask = results[key]
                mask = self._one_hot(mask)
                results[key] = mask

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}, '

        return repr_str

@TRANSFORMS.register_module()
class BinaryCls(BaseTransform):
    def __init__(self, ignore_index: int = 255):
        self.ignore_index = ignore_index

    def transform(self,
                  results: Dict):
        seg_map = results['gt_seg_map']
        assert len(seg_map.shape) == 2

        if self.ignore_index is not None:
            seg_map[seg_map == self.ignore_index] = 0

        label_cls = np.max(seg_map)

        results['cls'] = label_cls

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__

        return repr_str



@TRANSFORMS.register_module()
class BlockDCT(BaseTransform):
    def __init__(self, block_size=8, zigzag=True, channel='Y', shift=False, use_normalize=False, key=None):
        self.block_size = block_size
        self.zigzag = zigzag
        self.shift = shift
        self.use_normalize = use_normalize
        self.dct_idxs = self._get_dct_vector_coords(block_size)
        dct_encode_idx = self.dct_idxs.transpose(1, 0)
        self.zigzag_encode_idx = (dct_encode_idx[0], dct_encode_idx[1])

        self.c_table = {'Y': 0, 'U': 1, 'V': 2}
        self.idx_c = self.c_table[channel]

        self.key = key

    def transform(self,
                  results: dict):
        assert 'img' in results

        if self.key is not None:
            img = results[self.key]
        else:
            img = results['img']


        img_Y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img_Y = img_Y[:, :, self.idx_c]

        H, W = img_Y.shape[:2]

        if (H % self.block_size != 0) or (W % self.block_size != 0):
            img_Y = self._pad_img(img_Y)
            H, W = img_Y.shape[:2]

        # reshape to block
        img_Y = img_Y.reshape(H // self.block_size, self.block_size, W // self.block_size, self.block_size).transpose(0, 2, 1, 3)
        img_Y = img_Y.reshape(-1, self.block_size, self.block_size)


        img_Y = img_Y.astype(np.float32)
        if self.shift:
            img_Y = img_Y - 128

        # apply block dct
        dct = []

        for i in range(img_Y.shape[0]):
            dct_block = self._block_dct(img_Y[i])
            if self.zigzag:
                dct_block =  dct_block[self.zigzag_encode_idx]
            dct.append(dct_block)

        # reshape back
        dct = np.array(dct)

        dct = dct.reshape(H // self.block_size, W // self.block_size, self.block_size, self.block_size).transpose(0, 2, 1, 3)
        dct = dct.reshape(H, W)

        if self.use_normalize:
            dct = dct / 16320

        if self.key is not None:
            dct_key = self.key + '_dct'
            results[dct_key] = dct
        else:
            results['dct'] = dct

        return results

    def _pad_img(self, img):
        return mmcv.impad_to_multiple(img, divisor=self.block_size, pad_val=0)

    def _block_dct(self, block):
        return cv2.dct(block)

    def _get_dct_vector_coords(self, r=8):
        """
                Get the coordinates with zigzag order.
        """
        dct_index = []
        for i in range(r):
            if i % 2 == 0:  # start with even number
                index = [(i - j, j) for j in range(i + 1)]
                dct_index.extend(index)
            else:
                index = [(j, i - j) for j in range(i + 1)]
                dct_index.extend(index)
        for i in range(r, 2 * r - 1):
            if i % 2 == 0:
                index = [(i - j, j) for j in range(i - r + 1, r)]
                dct_index.extend(index)
            else:
                index = [(j, i - j) for j in range(i - r + 1, r)]
                dct_index.extend(index)
        dct_idxs = np.asarray(dct_index)
        return dct_idxs

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(block_size={self.block_size}, '

        return repr_str


@TRANSFORMS.register_module()
class DynamicResize(BaseTransform):
    def __init__(self, max_size,
                 resize_type=dict(type='Resize', keep_ratio=True),
                 keep_ratio=True):
        self.keep_ratio = keep_ratio

        self.scale = (int(4 * max_size), int(max_size))
        self.max_size = max_size

        if resize_type is not None:
            resize_type['scale'] = self.scale

        if resize_type is None:
            self.resize = TRANSFORMS.build({
                'type': 'Resize',
                'scale': self.scale,
                'keep_ratio': True
            })
        else:
            self.resize = TRANSFORMS.build(resize_type)

    def transform(self, results: dict):
        img = results['img']
        h, w = img.shape[:2]

        if min(h, w) > self.max_size:
            results = self.resize(results)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(max_size={self.max_size}, '
        repr_str += f'(keep_ratio={self.keep_ratio})'
        return repr_str


@TRANSFORMS.register_module()
class RandomRotate90(BaseTransform):
    """Rotate the image & seg.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - gt_seg_map


    """

    def __init__(self,
                 prob,
                 pad_val=0,
                 seg_pad_val=255,
                 center=None,
                 auto_bound=True):
        self.prob = prob
        assert prob >= 0 and prob <= 1
        self.degree = [0, 90, 180, 270]
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.center = center
        self.auto_bound = auto_bound


    @cache_randomness
    def generate_degree(self):
        return np.random.rand() < self.prob, random.choice(self.degree)


    def transform(self, results: dict) -> dict:
        """Call function to rotate image, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """

        rotate, degree = self.generate_degree()
        if rotate:
            # rotate image
            results['img'] = mmcv.imrotate(
                results['img'],
                angle=degree,
                border_value=self.pad_val,
                center=self.center,
                auto_bound=self.auto_bound)

            # rotate segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imrotate(
                    results[key],
                    angle=degree,
                    border_value=self.seg_pad_val,
                    center=self.center,
                    auto_bound=self.auto_bound,
                    interpolation='nearest')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, ' \
                    f'center={self.center}, ' \
                    f'auto_bound={self.auto_bound})'
        return repr_str


@TRANSFORMS.register_module()
class ELA(BaseTransform):
    """Rotate the image & seg.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - gt_seg_map


    """

    def __init__(self,
                 quality=90,
                 backend='cv2',
                 save_compression=False):
        self.quality = quality
        self.backend = backend
        self.save_compression = save_compression
        assert backend in ['pillow', 'cv2']

    def transform(self, results: dict) -> dict:
        """Call function to rotate image, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """
        img = results['img']
        h, w = img.shape[:2]

        if self.backend == 'cv2':
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
            result, encimg = cv2.imencode('.jpg', img, encode_param)
            decimg = cv2.imdecode(encimg, 1)
        else:
            with tempfile.NamedTemporaryFile(delete=True, suffix='.jpg') as tmp:
                im = Image.fromarray(img)
                im.save(tmp.name, "JPEG", quality=self.quality)
                decimg = Image.open(tmp.name)
                decimg = np.array(decimg)

        if self.save_compression:
            results['com_img'] = decimg

        results['ela'] = cv2.absdiff(img, decimg)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(quality={self.quality}, ' \
                    f'backend={self.backend}, ' \
                    f'save_compression={self.save_compression})'
        return repr_str


@TRANSFORMS.register_module()
class SubtractData(BaseTransform):
    def __init__(self, key1, key2, sub_key=None):
        self.key1 = key1
        self.key2 = key2
        self.sub_key = sub_key
        if sub_key is None:
            self.sub_key = 'diff'

    def transform(self, results: dict) -> dict:
        diff = cv2.absdiff(results[self.key1], results[self.key2])
        results[self.sub_key] = diff
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(key1={self.key1}, ' \
                    f'(key2={self.key2}, ' \
                    f'sub_key={self.sub_key})'
        return repr_str


@TRANSFORMS.register_module()
class AssignValue(BaseTransform):
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def transform(self, results: dict) -> dict:
        assert self.source in results
        results[self.target] = results[self.source]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(source={self.source}, ' \
                    f'target={self.target})'
        return repr_str

