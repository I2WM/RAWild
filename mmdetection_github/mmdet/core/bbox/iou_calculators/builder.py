from mmdet.registry import TASK_UTILS

IOU_CALCULATORS = TASK_UTILS


def _merge_default_args(cfg, default_args=None):
    if default_args is None or not isinstance(cfg, dict):
        return cfg
    cfg = cfg.copy()
    for k, v in default_args.items():
        cfg.setdefault(k, v)
    return cfg


def build_iou_calculator(cfg, **default_args):
    return TASK_UTILS.build(_merge_default_args(cfg, default_args))
