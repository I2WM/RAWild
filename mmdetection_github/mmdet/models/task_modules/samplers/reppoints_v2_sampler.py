import torch

from mmdet.registry import TASK_UTILS
from mmdet.models.task_modules.samplers.base_sampler import BaseSampler
from mmdet.models.task_modules.samplers.sampling_result import SamplingResult


@TASK_UTILS.register_module()
class RepPointsV2PseudoSampler(BaseSampler):
    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes, **kwargs):
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags)
        sampling_result.assign_result = assign_result
        return sampling_result
