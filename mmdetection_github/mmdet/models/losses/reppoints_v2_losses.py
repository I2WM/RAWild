import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.core import bbox_overlaps
from mmdet.models.losses.utils import weight_reduce_loss, weighted_loss
from mmdet.utils.rawild_debug_trace import maybe_dump_nonfinite


@weighted_loss
def quality_focal_loss(pred, target, beta=2.0, background_label=None):
    label, score = target
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction='none') * scale_factor.pow(beta)

    if background_label is None:
        bg_class_ind = pred.size(1)
        pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    else:
        pos = (label != background_label).nonzero().squeeze(1)
    pos_label = label[pos].long()
    if len(pos) > 0:
        scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
        loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
            pred[pos, pos_label], score[pos],
            reduction='none') * scale_factor.abs().pow(beta)

    return loss.sum(dim=1, keepdim=False)


def separate_sigmoid_focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25, reduction='mean', avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pos_inds = target.eq(1)
    neg_inds = target.lt(1)
    pos_weights = weight[pos_inds]
    pos_pred = pred_sigmoid[pos_inds]
    neg_pred = pred_sigmoid[neg_inds]
    pos_loss = -torch.log(pos_pred.clamp_min(1e-12)) * torch.pow(1 - pos_pred, gamma) * pos_weights * alpha
    neg_loss = -torch.log((1 - neg_pred).clamp_min(1e-12)) * torch.pow(neg_pred, gamma) * (1 - alpha)
    if pos_pred.nelement() == 0:
        loss = neg_loss.sum() / avg_factor
    else:
        loss = pos_loss.sum() / pos_weights.sum().clamp_min(1e-12) + neg_loss.sum() / avg_factor
    return loss


def giou_loss_oldver(pred, target, eps=1e-7):
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt + 1).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]
    ap = (pred[:, 2] - pred[:, 0] + 1) * (pred[:, 3] - pred[:, 1] + 1)
    ag = (target[:, 2] - target[:, 0] + 1) * (target[:, 3] - target[:, 1] + 1)
    union = ap + ag - overlap + eps
    ious = overlap / union
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1 + 1).clamp(min=0)
    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1] + eps
    gious = ious - (enclose_area - union) / enclose_area
    return 1 - gious


@MODELS.register_module()
class RPDQualityFocalLoss(nn.Module):
    def __init__(self, use_sigmoid=True, beta=2.0, reduction='mean', loss_weight=1.0):
        super().__init__()
        assert use_sigmoid is True
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.requires_box = True

    def forward(self, pred, target, weight=None, bbox_pred=None, bbox_target=None, bbox_weight=None, background_label=0, avg_factor=None, reduction_override=None):
        iou_score = bbox_target.new_zeros(target.size()).squeeze()
        pos_inds = (target != background_label).nonzero().squeeze(1)
        if len(pos_inds) > 0:
            score = bbox_overlaps(bbox_pred[pos_inds], bbox_target[pos_inds], is_aligned=True)
            iou_score[pos_inds] = score.detach()
        iou_score = iou_score.detach()
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * quality_focal_loss(
            pred,
            (target, iou_score),
            weight,
            beta=self.beta,
            background_label=background_label,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_cls


@MODELS.register_module()
class RPDGIoULoss(nn.Module):
    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        maybe_dump_nonfinite(
            'loss.giou.inputs',
            {'pred': pred, 'target': target, 'weight': weight},
            extra=dict(avg_factor=avg_factor, reduction=reduction),
            raise_on_nonfinite=True)
        loss = giou_loss_oldver(pred, target, eps=self.eps)
        maybe_dump_nonfinite(
            'loss.giou.raw_loss',
            {'loss': loss},
            extra=dict(avg_factor=avg_factor, reduction=reduction),
            raise_on_nonfinite=True)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        maybe_dump_nonfinite(
            'loss.giou.reduced_loss',
            {'loss': loss},
            extra=dict(avg_factor=avg_factor, reduction=reduction),
            raise_on_nonfinite=True)
        return self.loss_weight * loss


@MODELS.register_module()
class SEPFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        reduction = reduction_override if reduction_override else self.reduction
        return self.loss_weight * separate_sigmoid_focal_loss(pred, target, weight, gamma=self.gamma, alpha=self.alpha, reduction=reduction, avg_factor=avg_factor)
