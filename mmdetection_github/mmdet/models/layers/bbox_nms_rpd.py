import torch
from mmcv.ops.nms import batched_nms
from mmdet.structures.bbox import bbox_overlaps


def multiclass_nms_rpd(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None, inst_inds=None):
    num_classes = multi_scores.size(1) - 1
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
        if inst_inds is not None:
            inst_inds = inst_inds.view(inst_inds.size(0), -1)
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)
        if inst_inds is not None:
            inst_inds = inst_inds[:, None].expand(-1, num_classes)
    scores = multi_scores[:, :-1]
    if inst_inds is not None:
        inst_inds = inst_inds[scores > score_thr]
    labels = torch.arange(num_classes, dtype=torch.long, device=multi_scores.device)
    labels = labels.view(1, -1).expand_as(scores)
    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    if not torch.onnx.is_in_onnx_export():
        valid_mask = scores > score_thr
    if score_factors is not None:
        score_factors = score_factors.view(-1, 1).expand(multi_scores.size(0), num_classes).reshape(-1)
        scores = scores * score_factors
    if not torch.onnx.is_in_onnx_export():
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    else:
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)
    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS as it has not been executed this time')
        if inst_inds is not None:
            return bboxes, labels, inst_inds, None
        return bboxes, labels, None
    dets, keep = batched_nms(bboxes, scores.to(bboxes.dtype), labels, nms_cfg)
    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]
    if inst_inds is not None:
        return dets, labels[keep], inst_inds[keep], keep
    return dets, labels[keep], keep
