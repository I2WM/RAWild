from typing import Sequence

import numpy as np
import torch
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform

from mmdet.datasets.transforms.formatting import PackDetInputs
from mmdet.registry import TRANSFORMS
from mmdet.utils.rawild_debug_trace import maybe_dump_nonfinite


@TRANSFORMS.register_module(force=True)
class LoadRPDV2Annotations(BaseTransform):
    def __init__(self, num_classes: int, downsample: int = 8):
        self.num_classes = int(num_classes)
        self.downsample = int(downsample)

    def transform(self, results: dict) -> dict:
        gt_bboxes = results.get('gt_bboxes', None)
        gt_labels = results.get('gt_bboxes_labels', None)
        if gt_bboxes is None or gt_labels is None:
            return results
        if hasattr(gt_bboxes, 'tensor'):
            gt_bboxes = gt_bboxes.tensor
        gt_bboxes = np.asarray(gt_bboxes)
        gt_labels = np.asarray(gt_labels)
        pad_shape = results.get('pad_shape', results['img'].shape)
        h = int(pad_shape[0] / self.downsample)
        w = int(pad_shape[1] / self.downsample)
        gt_sem_map = np.zeros((self.num_classes, h, w), dtype=np.float32)
        gt_sem_weights = np.zeros((self.num_classes, h, w), dtype=np.float32)
        gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        if len(gt_bboxes) > 0:
            indexs = np.argsort(gt_areas)
            for ind in indexs[::-1]:
                cls_id = int(gt_labels[ind])
                if cls_id < 0 or cls_id >= self.num_classes:
                    continue
                box = gt_bboxes[ind]
                box_mask = np.zeros((h, w), dtype=np.int64)
                box_mask[int(box[1] / self.downsample):int(box[3] / self.downsample) + 1,
                         int(box[0] / self.downsample):int(box[2] / self.downsample) + 1] = 1
                gt_sem_map[cls_id][box_mask > 0] = 1
                area = max(float(gt_areas[ind]), 1e-6)
                gt_sem_weights[cls_id][box_mask > 0] = 1.0 / area
        results['gt_sem_map'] = gt_sem_map
        results['gt_sem_weights'] = gt_sem_weights
        maybe_dump_nonfinite(
            'transform.rpdv2_annotations',
            dict(
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_sem_map=gt_sem_map,
                gt_sem_weights=gt_sem_weights),
            extra=dict(
                img_path=results.get('img_path'),
                img_shape=results.get('img_shape'),
                pad_shape=pad_shape,
                downsample=self.downsample,
                num_classes=self.num_classes,
                gt_count=int(len(gt_bboxes)),
                gt_area_min=float(gt_areas.min()) if len(gt_areas) > 0 else None,
                gt_area_max=float(gt_areas.max()) if len(gt_areas) > 0 else None),
            raise_on_nonfinite=True)
        return results


@TRANSFORMS.register_module(force=True)
class RPDV2FormatBundle(PackDetInputs):
    def __init__(self, meta_keys: Sequence[str]):
        super().__init__(meta_keys=meta_keys)

    def transform(self, results: dict) -> dict:
        packed = super().transform(results)
        data_sample = packed['data_samples']
        if 'gt_sem_map' in results:
            data_sample.set_field(to_tensor(results['gt_sem_map']), 'gt_sem_map', dtype=torch.Tensor)
        if 'gt_sem_weights' in results:
            data_sample.set_field(to_tensor(results['gt_sem_weights']), 'gt_sem_weights', dtype=torch.Tensor)
        return packed
