import torch

from ..builder import BBOX_ASSIGNERS
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class PointAssignerV2(BaseAssigner):
    def __init__(self, scale=4, pos_num=3, mask_center_sample=False, use_center=False):
        self.scale = scale
        self.pos_num = pos_num
        self.mask_center_sample = mask_center_sample
        self.use_center = use_center

    def assign(self, points, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None, gt_bit_masks=None):
        INF = 1e8
        num_gts, num_points = gt_bboxes.shape[0], points.shape[0]
        if num_gts == 0 or num_points == 0:
            assigned_gt_inds = points.new_full((num_points,), 0, dtype=torch.long)
            assigned_labels = None if gt_labels is None else points.new_full((num_points,), -1, dtype=torch.long)
            return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
        points_xy = points[:, :2]
        points_stride = points[:, 2]
        points_lvl = torch.log2(points_stride).int()
        lvl_min, lvl_max = points_lvl.min(), points_lvl.max()
        gt_bboxes_xy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
        if self.mask_center_sample:
            _, _h, _w = gt_bit_masks.size()
            _ys = torch.arange(0, _h, dtype=torch.float32, device=gt_bit_masks.device)
            _xs = torch.arange(0, _w, dtype=torch.float32, device=gt_bit_masks.device)
            m00 = gt_bit_masks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
            m10 = (gt_bit_masks * _xs).sum(dim=-1).sum(dim=-1)
            m01 = (gt_bit_masks * _ys[:, None]).sum(dim=-1).sum(dim=-1)
            center_xs = m10 / m00
            center_ys = m01 / m00
            gt_center_xy = torch.stack((center_xs, center_ys), dim=-1)
        else:
            gt_center_xy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
        gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).clamp(min=1e-6)
        scale = self.scale
        gt_bboxes_lvl = ((torch.log2(gt_bboxes_wh[:, 0] / scale) + torch.log2(gt_bboxes_wh[:, 1] / scale)) / 2).int()
        gt_bboxes_lvl = torch.clamp(gt_bboxes_lvl, min=lvl_min, max=lvl_max)
        if self.use_center:
            distances = ((points_xy[:, None, :] - gt_center_xy[None, :, :]) / gt_bboxes_wh[None, :, :]).norm(dim=2)
        else:
            distances = ((points_xy[:, None, :] - gt_bboxes_xy[None, :, :]) / gt_bboxes_wh[None, :, :]).norm(dim=2)
        distances[points_lvl[:, None] != gt_bboxes_lvl[None, :]] = INF
        assigned_gt_inds = points.new_zeros((num_points,), dtype=torch.long)
        min_dist, min_dist_index = torch.topk(distances, self.pos_num, dim=0, largest=False)
        distances_inf = torch.full_like(distances, INF)
        distances_inf[min_dist_index, torch.arange(num_gts)] = min_dist
        min_dist, min_dist_index = distances_inf.min(dim=1)
        assigned_gt_inds[min_dist != INF] = min_dist_index[min_dist != INF] + 1
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_points,), -1)
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)


