from .builder import build_iou_calculator
from mmdet.models.task_modules.assigners.iou2d_calculator import BboxOverlaps2D, bbox_overlaps

__all__ = [ build_iou_calculator, BboxOverlaps2D, bbox_overlaps]
