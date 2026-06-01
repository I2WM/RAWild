from mmdet.registry import MODELS

BACKBONES = MODELS
NECKS = MODELS
ROI_EXTRACTORS = MODELS
SHARED_HEADS = MODELS
HEADS = MODELS
LOSSES = MODELS
DETECTORS = MODELS


def _merge_default_args(cfg, default_args=None):
    if default_args is None or not isinstance(cfg, dict):
        return cfg
    cfg = cfg.copy()
    for k, v in default_args.items():
        cfg.setdefault(k, v)
    return cfg


def build(cfg, registry=None, default_args=None):
    if isinstance(cfg, (list, tuple)):
        return [build(c, registry=registry, default_args=default_args) for c in cfg]
    return MODELS.build(_merge_default_args(cfg, default_args))


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_neck(cfg):
    return build(cfg, NECKS)


def build_roi_extractor(cfg):
    return build(cfg, ROI_EXTRACTORS)


def build_shared_head(cfg):
    return build(cfg, SHARED_HEADS)


def build_head(cfg):
    return build(cfg, HEADS)


def build_loss(cfg):
    return build(cfg, LOSSES)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    return build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
