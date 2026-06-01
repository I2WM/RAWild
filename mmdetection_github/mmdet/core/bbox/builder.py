from mmdet.registry import TASK_UTILS

BBOX_ASSIGNERS = TASK_UTILS
BBOX_SAMPLERS = TASK_UTILS
BBOX_CODERS = TASK_UTILS


def _merge_default_args(cfg, default_args=None):
    if default_args is None or not isinstance(cfg, dict):
        return cfg
    cfg = cfg.copy()
    for k, v in default_args.items():
        cfg.setdefault(k, v)
    return cfg


def build_assigner(cfg, **default_args):
    return TASK_UTILS.build(_merge_default_args(cfg, default_args))


def build_sampler(cfg, **default_args):
    return TASK_UTILS.build(_merge_default_args(cfg, default_args))


def build_bbox_coder(cfg, **default_args):
    return TASK_UTILS.build(_merge_default_args(cfg, default_args))
