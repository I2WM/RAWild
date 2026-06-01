from typing import List, Tuple

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import get_box_tensor
from mmdet.utils.rawild_debug_trace import (maybe_dump_nonfinite,
                                            register_gradient_hooks,
                                            summarize_img_metas)

from .single_stage import SingleStageDetector


def _pad_and_stack_semantic_targets(tensors: List[torch.Tensor],
                                    device: torch.device) -> torch.Tensor:
    max_h = max(int(t.shape[-2]) for t in tensors)
    max_w = max(int(t.shape[-1]) for t in tensors)
    padded = []
    for tensor in tensors:
        pad_h = max_h - int(tensor.shape[-2])
        pad_w = max_w - int(tensor.shape[-1])
        if pad_h or pad_w:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=0)
        padded.append(tensor)
    return torch.stack(padded, dim=0).to(device)


@MODELS.register_module()
class RepPointsV2Detector(SingleStageDetector):
    def loss(self, batch_inputs: torch.Tensor, batch_data_samples: SampleList):
        trace_extra = dict(
            img_metas=summarize_img_metas([s.metainfo for s in batch_data_samples]),
            gt_bbox_counts=[
                int(get_box_tensor(s.gt_instances.bboxes).size(0))
                for s in batch_data_samples
            ],
            gt_label_counts=[
                int(s.gt_instances.labels.size(0)) for s in batch_data_samples
            ])
        maybe_dump_nonfinite(
            'detector.loss.batch_inputs',
            {'batch_inputs': batch_inputs},
            extra=trace_extra,
            raise_on_nonfinite=True)
        x = self.extract_feat(batch_inputs, batch_data_samples)
        feat_tensors = {f'feat_{idx}': feat for idx, feat in enumerate(x)}
        maybe_dump_nonfinite(
            'detector.loss.features',
            feat_tensors,
            extra=trace_extra,
            raise_on_nonfinite=True)
        register_gradient_hooks('detector.features', feat_tensors, extra=trace_extra)
        outs = self.bbox_head(x)
        out_tensors = {}
        branch_names = (
            'cls_scores', 'pts_preds_init', 'pts_preds_refine',
            'hm_scores', 'hm_offsets', 'sem_scores')
        for branch_name, branch_value in zip(branch_names, outs):
            for idx, tensor in enumerate(branch_value):
                out_tensors[f'{branch_name}[{idx}]'] = tensor
        maybe_dump_nonfinite(
            'detector.loss.head_outputs',
            out_tensors,
            extra=trace_extra,
            raise_on_nonfinite=True)
        register_gradient_hooks('detector.head_outputs', out_tensors, extra=trace_extra)
        gt_bboxes = [get_box_tensor(s.gt_instances.bboxes) for s in batch_data_samples]
        gt_labels = [s.gt_instances.labels for s in batch_data_samples]
        img_metas = [s.metainfo for s in batch_data_samples]
        if hasattr(batch_data_samples[0], 'gt_rpd_sem'):
            gt_sem_map = _pad_and_stack_semantic_targets(
                [s.gt_rpd_sem.sem_map for s in batch_data_samples],
                batch_inputs.device)
            gt_sem_weights = _pad_and_stack_semantic_targets(
                [s.gt_rpd_sem.sem_weights for s in batch_data_samples],
                batch_inputs.device)
        else:
            gt_sem_map = _pad_and_stack_semantic_targets(
                [s.gt_sem_map for s in batch_data_samples],
                batch_inputs.device)
            gt_sem_weights = _pad_and_stack_semantic_targets(
                [s.gt_sem_weights for s in batch_data_samples],
                batch_inputs.device)
        has_ignore = any(hasattr(s, 'ignored_instances') and hasattr(s.ignored_instances, 'bboxes') and len(s.ignored_instances.bboxes) > 0 for s in batch_data_samples)
        gt_bboxes_ignore = None
        if has_ignore:
            gt_bboxes_ignore = []
            for s in batch_data_samples:
                if hasattr(s, 'ignored_instances') and hasattr(s.ignored_instances, 'bboxes') and len(s.ignored_instances.bboxes) > 0:
                    gt_bboxes_ignore.append(get_box_tensor(s.ignored_instances.bboxes))
                else:
                    gt_bboxes_ignore.append(batch_inputs.new_zeros((0, 4)))
        losses = self.bbox_head.loss(*outs, gt_bboxes, gt_sem_map, gt_sem_weights, gt_labels, img_metas, gt_bboxes_ignore=gt_bboxes_ignore)
        maybe_dump_nonfinite(
            'detector.loss.loss_dict',
            losses,
            extra=trace_extra,
            raise_on_nonfinite=True)
        return losses

    def predict(self, batch_inputs: torch.Tensor, batch_data_samples: SampleList, rescale: bool = True):
        x = self.extract_feat(batch_inputs, batch_data_samples)
        outs = self.bbox_head(x)
        img_metas = [s.metainfo for s in batch_data_samples]
        results = self.bbox_head.get_bboxes(*outs, img_metas, cfg=self.test_cfg, rescale=rescale, nms=True)
        results_list = []
        for det_bboxes, det_labels in results:
            pred_instances = InstanceData()
            pred_instances.bboxes = det_bboxes[:, :4]
            if det_bboxes.numel() == 0:
                pred_instances.scores = det_bboxes.new_zeros((0,))
            elif det_bboxes.shape[1] >= 5:
                pred_instances.scores = det_bboxes[:, 4]
            else:
                pred_instances.scores = det_bboxes.new_ones((det_bboxes.shape[0],))
            pred_instances.labels = det_labels
            results_list.append(pred_instances)
        return self.add_pred_to_datasample(batch_data_samples, results_list)

    def _forward(self, batch_inputs: torch.Tensor, batch_data_samples=None) -> Tuple[List[torch.Tensor]]:
        x = self.extract_feat(batch_inputs, batch_data_samples)
        return self.bbox_head(x)
