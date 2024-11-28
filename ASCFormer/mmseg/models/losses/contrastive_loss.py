# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


SMALL_NUM = np.log(1e-45)



@MODELS.register_module()
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07,
                 loss_weight=1.0,
                 gather=False,
                 min_points=None):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.loss_weight = loss_weight
        self.gather = gather
        self.min_points = min_points
        if min_points is None:
            self.min_points = 2

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = features.device
        current_rank = torch.distributed.get_rank()
        if self.gather:
            assert mask is None

            all_features = get_all_gather_with_various_shape(features)
            # for i in range(len(all_features)):
            #     all_features[i].requires_grad=True
            all_features[current_rank] = features
            features = torch.cat(all_features, dim=0)

            if labels is not None:
                all_labels = get_all_gather_with_various_shape(labels)
                all_labels[current_rank] = labels
                labels = torch.cat(all_labels, dim=0)


        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        num_T = labels.sum()

        if num_T < self.min_points or (batch_size - self.min_points) < 2:
            # print("construct sample pairs failed")
            return None

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        loss = self.loss_weight * loss

        return loss



@MODELS.register_module()
class SupDCLLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.1, contrast_mode='one',
                 base_temperature=0.07, loss_weight=1.0):
        super(SupDCLLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.loss_weight = loss_weight

    def forward(self, pos_feats, neg_feats):
        """Compute loss for model.

        Args:
            pos_feats: hidden vector of shape [M, C].
            neg_feats: ground truth of shape [N, C].

        Returns:
            A loss scalar.
        """
        pos_feats = concat_all_gather(pos_feats)
        neg_feats = concat_all_gather(neg_feats)

        print(pos_feats.shape)

        M, C = pos_feats.shape
        N = neg_feats.shape[0]

        pos_feats = F.normalize(pos_feats, dim=1)
        neg_feats = F.normalize(neg_feats, dim=1)

        # compute logits
        pos1_simi = torch.div(torch.matmul(pos_feats, pos_feats.T), self.temperature)
        pos2_simi = torch.div(torch.matmul(neg_feats, neg_feats.T), self.temperature)

        neg_simi = torch.div(torch.matmul(pos_feats, neg_feats.T), self.temperature)

        exp_pos1 = torch.exp(pos1_simi)
        exp_pos2 = torch.exp(pos2_simi)
        exp_neg = torch.exp(neg_simi)

        diag_mask1 = torch.eye(M, device=exp_pos1.device, dtype=torch.bool)
        diag_mask2 = torch.eye(N, device=exp_pos2.device, dtype=torch.bool)

        exp_neg1 = torch.sum(exp_neg, dim=1, keepdim=True)
        exp_neg2 = torch.sum(exp_neg.transpose(0, 1), dim=1, keepdim=True)

        exp_neg1 = exp_pos1 + exp_neg1
        exp_neg2 = exp_pos2 + exp_neg2

        # DCL loss
        exp_pos1 = -torch.log(torch.div(exp_pos1, exp_neg1))
        exp_pos2 = -torch.log(torch.div(exp_pos2, exp_neg2))

        exp_pos1 = exp_pos1[~diag_mask1]
        exp_pos2 = exp_pos2[~diag_mask2]

        # SupDCL loss
        loss1 = torch.mean(exp_pos1)
        loss2 = torch.mean(exp_pos2)
        if loss1 < 0 or loss2 < 0:
            print("warning: pos-{}, neg-{}, loss1: {}, loss2: {}".format(M, N, loss1, loss2))
        loss = 0.5 * (loss1 + loss2) * self.loss_weight

        return loss


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


@torch.no_grad()
def get_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """

    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    return tensors_gather


@torch.no_grad()
def get_all_gather_with_various_shape(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """

    tensor_size = torch.tensor(tensor.size()).to(tensor.device)
    device = tensor.device
    dtype = tensor.dtype

    size_gather = [torch.zeros_like(tensor_size) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(size_gather, tensor_size, async_op=False)

    tensors_gather = [torch.ones(torch.Size(_size), dtype=dtype).to(device) for _size in size_gather]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    return tensors_gather



@torch.no_grad()
def concat_other_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    current_rank = torch.distributed.get_rank()

    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    tensors_gather.pop(current_rank)
    output = torch.cat(tensors_gather, dim=0)
    return output


