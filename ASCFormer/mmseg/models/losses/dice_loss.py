# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/LikeLy-Journey/SegmenTron/blob/master/
segmentron/solver/loss.py (Apache-2.0 License)"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .utils import get_class_weight, weighted_loss


@weighted_loss
def dice_loss(pred,
              target,
              valid_mask,
              smooth=1,
              exponent=2,
              class_weight=None,
              ignore_index=255):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            dice_loss = binary_dice_loss(
                pred[:, i],
                target[..., i],
                valid_mask=valid_mask,
                smooth=smooth,
                exponent=exponent)
            if class_weight is not None:
                dice_loss *= class_weight[i]
            total_loss += dice_loss
    return total_loss / num_classes


@weighted_loss
def binary_dice_loss(pred, target, valid_mask, smooth=1, exponent=2, **kwards):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth

    return 1 - num / den


@MODELS.register_module()
class DiceLoss(nn.Module):
    """DiceLoss.

    This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.

    Args:
        smooth (float): A float number to smooth loss, and avoid NaN error.
            Default: 1
        exponent (float): An float number to calculate denominator
            value: \\sum{x^exponent} + \\sum{y^exponent}. Default: 2.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        ignore_index (int | None): The label index to be ignored. Default: 255.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_dice'.
    """

    def __init__(self,
                 smooth=1,
                 exponent=2,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_dice',
                 **kwards):
        super().__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                avg_factor=None,
                reduction_override=None,
                **kwards):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()

        loss = self.loss_weight * dice_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            ignore_index=self.ignore_index)
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


@MODELS.register_module()
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, exponent=2, reduction='mean',
                 loss_weight=1.0,
                 # ignore_index=255,
                 loss_name='loss_dice',
                 ):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction

        self.loss_weight = loss_weight
        # self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.exponent) + target.pow(self.exponent), dim=1) + self.smooth

        loss = 1 - num / den
        loss = self.loss_weight * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


