import os
from copy import deepcopy

default_scope = 'mmdet'

DATA_ROOT = os.environ.get('RAWILD_DATA_ROOT', 'data/RAWild_datasets')
PASCAL_NPY_ROOT = os.environ.get(
    'RAWILD_PASCAL_NPY_ROOT', f'{DATA_ROOT}/PASCAL_RAW_npy')
PRETRAINED_ROOT = os.environ.get('RAWILD_PRETRAINED_ROOT', 'checkpoints/pretrained')
RUNTIME_ROOT = os.environ.get('RAWILD_WORK_DIR_ROOT', 'work_dirs')
WORK_DIR_ROOT = f'{RUNTIME_ROOT}/RAWild_swint'
CHECKPOINTS = dict(
    swint_backbone=f'{PRETRAINED_ROOT}/reppointsv2_swin_tiny_patch4_window7_3x_backboneonly_20260501.pth',
    rod_detector=f'{PRETRAINED_ROOT}/reppointsv2_swin_tiny_patch4_window7_3x_rod5_detectorinit_20260430.pth',
    aodraw_detector=f'{PRETRAINED_ROOT}/reppointsv2_swin_tiny_patch4_window7_3x_aod62_darkisp_detectorinit_20260430.pth',
)

DATA_ROOTS = dict(
    pascal=f'{DATA_ROOT}/PASCAL_RAW',
    pascal_npy=PASCAL_NPY_ROOT,
    lod=f'{DATA_ROOT}/LOD_BMVC2021',
    rod=f'{DATA_ROOT}/ROD_dataset',
    aodraw=f'{DATA_ROOT}/AODRaw_dataset',
)

PASCAL_CLASSES = ('car', 'person', 'bicycle')
LOD_CLASSES = (
    'bicycle', 'car', 'motorbike', 'chair',
    'diningtable', 'bottle', 'tvmonitor', 'bus',
)
VOC_LOD_MIX_CLASSES = (
    'car', 'person', 'bicycle', 'motorbike', 'chair',
    'diningtable', 'bottle', 'tvmonitor', 'bus',
)
ROD_CLASSES = ('Tram', 'Car', 'Truck', 'Cyclist', 'Pedestrian')
AODRAW_CLASSES = (
    'person', 'traffic_sign', 'surveillance_camera', 'bicyle', 'car', 'tricycle',
    'truck', 'traffic_light', 'motorcycle', 'handbag/satchel', 'bottle/cup',
    'backpack', 'bus_stop_sign', 'helmet', 'garbage_can', 'bus', 'dog', 'hat',
    'chair', 'table', 'phone', 'refrigerator', 'traffic_cone', 'fire_hydrant',
    'crane', 'tent', 'fire_extinguisher', 'bowl', 'cat', 'sink', 'lamp',
    'monitor', 'bench', 'spoon', 'earphone', 'potted_plant', 'vase',
    'suitcase', 'vending_machine', 'watch', 'train', 'boat', 'umbrella',
    'sofa', 'plate', 'pot', 'pillow', 'scissors', 'mouse', 'desk_lamp',
    'keyboard', 'toilet_paper', 'pen', 'computer_box', 'laptop', 'mirror',
    'cans', 'bed', 'toilet', 'wine_glass', 'clock', 'airplane',
)

VOC_RESIZE = (667, 400)
COCO_RESIZE = (1333, 800)
MAX_EPOCHS = 20
LR_MILESTONES = [16]
PASCAL_MIX_REPEAT_TIMES = 1
LOD_MIX_REPEAT_TIMES = 8
PACK_META_KEYS = (
    'img_id', 'img_path', 'ori_shape', 'img_shape', 'pad_shape',
    'scale_factor', 'flip', 'flip_direction', 'raw_bit_depth',
)
BACKEND_ARGS = None
NORM_CFG = dict(type='GN', num_groups=32, requires_grad=True)

custom_imports = dict(
    imports=[
        'mmdet.datasets',
        'mmdet.datasets.pascal_raw',
        'mmdet.datasets.lod_bmvc21',
        'mmdet.datasets.aodraw',
        'mmdet.models.data_preprocessors.data_preprocessor',
        'mmdet.datasets.transforms.load_npy',
        'mmdet.datasets.transforms.load_png_dual',
        'mmdet.datasets.transforms.reppoints_v2_transforms',
        'mmdet.models.backbones.raw_swin_dia',
        'mmdet.models.necks.bifpn',
        'mmdet.models.layers.bbox_nms_rpd',
        'mmdet.models.task_modules.samplers.reppoints_v2_sampler',
        'mmdet.models.losses.gaussian_focal_loss',
        'mmdet.models.losses.reppoints_v2_losses',
        'mmdet.models.dense_heads.reppoints_v2_head',
        'mmdet.models.detectors.reppoints_v2_detector',
    ],
    allow_failed_imports=False,
)


def _copy(value):
    return deepcopy(value)


def _metainfo(classes):
    return dict(classes=classes, palette=None)


def _scale_tag(scale):
    return f'{scale[0]}x{scale[1]}'


def _model(num_classes):
    return dict(
        type='RepPointsV2Detector',
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            bgr_to_rgb=False,
            pad_size_divisor=32,
        ),
        backbone=dict(
            type='RAWildSwinTransformer',
            pretrain_img_size=224,
            in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads_swin=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.0,
            ape=False,
            drop_path_rate=0.2,
            use_checkpoint=False,
            patch_norm=True,
            out_indices=(1, 2, 3),
            grid_depth=8,
            transformer_dim=128,
            num_heads=4,
            num_layers=2,
            curve_n=6,
            use_bezier=True,
            use_grid=True,
            bezier_type='init_bias',
            dia_k=0.05,
            dia_activation='exp_tanh',
            dia_exp_alpha=1.0,
            dia_matrix_mode='d_ia',
            cond_injection_type='adaln',
            hist_bins=64,
            hist_sigma=0.1,
            hist_downsample_factor=16,
            hist_prefix_tokens=8,
            bit_embed_min=8,
            bit_embed_max=24,
            bit_depth_norm_max=24.0,
            frozen_stages=-1,
            pretrained=CHECKPOINTS['swint_backbone'],
        ),
        neck=dict(
            type='BiFPN',
            in_channels=[192, 384, 768],
            out_channels=256,
            start_level=0,
            add_extra_convs=False,
            num_outs=5,
            num_repeat=2,
            no_norm_on_lateral=False,
            norm_cfg=NORM_CFG,
        ),
        bbox_head=dict(
            type='RepPointsV2Head',
            num_classes=num_classes,
            in_channels=256,
            feat_channels=256,
            point_feat_channels=256,
            stacked_convs=3,
            shared_stacked_convs=1,
            first_kernel_size=3,
            kernel_size=1,
            corner_dim=64,
            num_points=9,
            gradient_mul=0.1,
            point_strides=[8, 16, 32, 64, 128],
            point_base_scale=4,
            norm_cfg=NORM_CFG,
            loss_cls=dict(type='RPDQualityFocalLoss', use_sigmoid=True, beta=2.0, loss_weight=1.0),
            loss_bbox_init=dict(type='RPDGIoULoss', loss_weight=1.0),
            loss_bbox_refine=dict(type='RPDGIoULoss', loss_weight=2.0),
            loss_heatmap=dict(type='GaussianFocalLoss', alpha=2.0, gamma=4.0, loss_weight=0.25),
            loss_offset=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
            loss_sem=dict(type='SEPFocalLoss', gamma=2.0, alpha=0.25, loss_weight=0.1),
            transform_method='exact_minmax',
        ),
        train_cfg=dict(
            init=dict(assigner=dict(type='PointAssignerV2', scale=4, pos_num=1), allowed_border=-1, pos_weight=-1, debug=False),
            heatmap=dict(assigner=dict(type='PointHMAssigner', gaussian_bump=True, gaussian_iou=0.7), allowed_border=-1, pos_weight=-1, debug=False),
            refine=dict(assigner=dict(type='ATSSAssignerV2', topk=9), allowed_border=-1, pos_weight=-1, debug=False),
        ),
        test_cfg=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=100,
        ),
    )


def _train_shell(batch_size=1, num_workers=2):
    return dict(
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=True),
        batch_sampler=dict(type='AspectRatioBatchSampler'),
    )


def _eval_shell():
    return dict(
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
    )


def _voc_loader(light_key):
    npy_root = {
        'pas_low': f"{DATA_ROOTS['pascal_npy']}/demosaic_low_12bit",
        'pas_nm': f"{DATA_ROOTS['pascal_npy']}/demosaic_normal_12bit",
        'pas_oe': f"{DATA_ROOTS['pascal_npy']}/demosaic_oe_12bit",
    }[light_key]
    return dict(type='LoadImageFromNpy', bits=12, img_subdir_npy=npy_root, dual_precision=True)


def _lod_loader():
    return dict(type='LoadImageFromPngDual', bits=16)


def _rod_loader():
    return dict(type='LoadImageFromNpy', bits=16, img_subdir_npy='', dual_precision=True)


def _aodraw_loader():
    return dict(type='LoadImageFromNpy', bits=8, img_subdir_npy='', dual_precision=True)


def _voc_train_pipeline(loader, resize_scale=VOC_RESIZE, num_classes=3):
    return [
        loader,
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', scale=resize_scale, keep_ratio=True),
        dict(type='RandomFlip', prob=0.5),
        dict(type='LoadRPDV2Annotations', num_classes=num_classes),
        dict(type='RPDV2FormatBundle', meta_keys=PACK_META_KEYS),
    ]


def _voc_test_pipeline(loader, resize_scale=VOC_RESIZE, num_classes=3):
    return [
        loader,
        dict(type='Resize', scale=resize_scale, keep_ratio=True),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='LoadRPDV2Annotations', num_classes=num_classes),
        dict(type='RPDV2FormatBundle', meta_keys=PACK_META_KEYS),
    ]


def _coco_train_pipeline(loader, resize_scale=COCO_RESIZE, num_classes=5):
    return [
        loader,
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', scale=resize_scale, keep_ratio=True),
        dict(type='RandomFlip', prob=0.5),
        dict(type='LoadRPDV2Annotations', num_classes=num_classes),
        dict(type='RPDV2FormatBundle', meta_keys=PACK_META_KEYS),
    ]


def _coco_test_pipeline(loader, resize_scale=COCO_RESIZE, num_classes=5):
    return [
        loader,
        dict(type='Resize', scale=resize_scale, keep_ratio=True),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='LoadRPDV2Annotations', num_classes=num_classes),
        dict(type='RPDV2FormatBundle', meta_keys=PACK_META_KEYS),
    ]


def _pascal_dataset(light_key, split, test_mode, classes=PASCAL_CLASSES,
                    resize_scale=VOC_RESIZE):
    img_subdir = {
        'pas_low': 'original/demosaic_low',
        'pas_nm': 'original/raw',
        'pas_oe': 'original/demosaic_oe',
    }[light_key]
    loader = _voc_loader(light_key)
    num_classes = len(classes)
    dataset = dict(
        type='PASCAL_RAW',
        data_root=DATA_ROOTS['pascal'],
        ann_file=f'trainval/{split}.txt',
        img_subdir=img_subdir,
        ann_subdir='annotations',
        data_prefix=dict(sub_data_root=''),
        metainfo=_metainfo(classes),
        test_mode=test_mode,
        pipeline=(
            _voc_test_pipeline(loader, resize_scale, num_classes)
            if test_mode else _voc_train_pipeline(loader, resize_scale, num_classes)
        ),
        backend_args=BACKEND_ARGS,
    )
    if not test_mode:
        dataset['filter_cfg'] = dict(filter_empty_gt=True, min_size=32, bbox_min_size=32)
    return dataset


def _lod_dataset(split, test_mode, classes=LOD_CLASSES,
                 resize_scale=VOC_RESIZE):
    loader = _lod_loader()
    num_classes = len(classes)
    dataset = dict(
        type='LOD_Dataset',
        data_root=DATA_ROOTS['lod'],
        ann_file=f'trainval/{split}.txt',
        img_subdir='RAW_dark',
        ann_subdir='RAW-dark-Annotations',
        data_prefix=dict(sub_data_root=''),
        metainfo=_metainfo(classes),
        test_mode=test_mode,
        pipeline=(
            _voc_test_pipeline(loader, resize_scale, num_classes)
            if test_mode else _voc_train_pipeline(loader, resize_scale, num_classes)
        ),
        backend_args=BACKEND_ARGS,
    )
    if not test_mode:
        dataset['filter_cfg'] = dict(filter_empty_gt=True, min_size=32, bbox_min_size=32)
    return dataset


def _rod_dataset(split, test_mode, resize_scale=COCO_RESIZE):
    loader = _rod_loader()
    return dict(
        type='CocoDataset',
        data_root=DATA_ROOTS['rod'],
        ann_file=f'annotations/{split}.json',
        data_prefix=dict(img=f'images/{split}/'),
        metainfo=_metainfo(ROD_CLASSES),
        test_mode=test_mode,
        pipeline=(
            _coco_test_pipeline(loader, resize_scale, len(ROD_CLASSES))
            if test_mode else _coco_train_pipeline(loader, resize_scale, len(ROD_CLASSES))
        ),
        backend_args=BACKEND_ARGS,
        filter_cfg=None if test_mode else dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
    )


def _aodraw_dataset(split, test_mode, resize_scale=COCO_RESIZE):
    loader = _aodraw_loader()
    return dict(
        type='AODRawDataset',
        data_root=DATA_ROOTS['aodraw'],
        ann_file=(
            'AODRaw/annotations/test_annotations_downsample_scale3_bbox_min_size32.json'
            if test_mode else
            'AODRaw/annotations/train_annotations_downsample_scale3_bbox_min_size32.json'
        ),
        data_prefix=dict(
            img='AODRaw_test/images_downsampled_raw/' if test_mode else 'AODRaw_train/images_downsampled_raw/'
        ),
        image_suffix='.npy',
        metainfo=_metainfo(AODRAW_CLASSES),
        test_mode=test_mode,
        pipeline=(
            _coco_test_pipeline(loader, resize_scale, len(AODRAW_CLASSES))
            if test_mode else _coco_train_pipeline(loader, resize_scale, len(AODRAW_CLASSES))
        ),
        backend_args=BACKEND_ARGS,
        filter_cfg=None if test_mode else dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
    )


def _common_runtime(max_epochs=MAX_EPOCHS):
    return dict(
        optim_wrapper=dict(
            type='OptimWrapper',
            optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005),
            paramwise_cfg=dict(custom_keys=dict(bilateral_grid_adapter=dict(lr_mult=10.0))),
            clip_grad=dict(max_norm=35, norm_type=2),
        ),
        param_scheduler=[
            dict(type='LinearLR', begin=0, by_epoch=False, end=500, start_factor=0.001),
            dict(type='MultiStepLR', begin=0, by_epoch=True, end=max_epochs, milestones=LR_MILESTONES, gamma=0.1),
        ],
        train_cfg=dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1),
        val_cfg=dict(type='ValLoop'),
        test_cfg=dict(type='TestLoop'),
        auto_scale_lr=dict(base_batch_size=4, enable=False),
        load_from=None,
        resume=False,
    )


def _voc_lod_mix_dataset(split, test_mode):
    lod_dataset = _lod_dataset(
        split, test_mode,
        classes=VOC_LOD_MIX_CLASSES, resize_scale=VOC_RESIZE)
    if not test_mode:
        lod_dataset = dict(
            type='RepeatDataset',
            times=LOD_MIX_REPEAT_TIMES,
            dataset=lod_dataset,
        )

    datasets = [
        _pascal_dataset(
            'pas_low', split, test_mode,
            classes=VOC_LOD_MIX_CLASSES, resize_scale=VOC_RESIZE),
        _pascal_dataset(
            'pas_nm', split, test_mode,
            classes=VOC_LOD_MIX_CLASSES, resize_scale=VOC_RESIZE),
        _pascal_dataset(
            'pas_oe', split, test_mode,
            classes=VOC_LOD_MIX_CLASSES, resize_scale=VOC_RESIZE),
        lod_dataset,
    ]
    return dict(
        type='ConcatDataset',
        datasets=datasets,
        ignore_keys=['dataset_type', 'palette'],
    )


def build_pascal_config(light_key):
    cfg = _common_runtime()
    cfg.update(
        experiment_config_name=f'RAWild_swint_{light_key}',
        model=_model(len(PASCAL_CLASSES)),
        train_dataloader=_train_shell(),
        val_dataloader=_eval_shell(),
        test_dataloader=_eval_shell(),
        val_evaluator=dict(type='VOCMetric', metric='mAP', iou_thrs=[0.5, 0.75], eval_mode='11points'),
        test_evaluator=dict(type='VOCMetric', metric='mAP', iou_thrs=[0.5, 0.75], eval_mode='11points'),
        work_dir=f'{WORK_DIR_ROOT}/{light_key}',
    )
    cfg['model']['backbone'].update(bit_depth_norm_max=12.0)
    cfg['train_dataloader']['dataset'] = dict(
        type='RepeatDataset',
        times=5,
        dataset=_pascal_dataset(light_key, 'train', False),
    )
    cfg['val_dataloader']['dataset'] = _pascal_dataset(light_key, 'val', True)
    cfg['test_dataloader']['dataset'] = _pascal_dataset(light_key, 'val', True)
    return cfg


def build_pascal_lod_mix_config(max_epochs=MAX_EPOCHS, train_batch_size=1):
    batch_suffix = '' if train_batch_size == 1 else f'_bs{train_batch_size}'
    cfg = _common_runtime(max_epochs=max_epochs)
    cfg.update(
        experiment_config_name=f'RAWild_swint_pascal_lod_mix_667x400{batch_suffix}_epoch{max_epochs}',
        model=_model(len(VOC_LOD_MIX_CLASSES)),
        train_dataloader=_train_shell(batch_size=train_batch_size),
        val_dataloader=_eval_shell(),
        test_dataloader=_eval_shell(),
        val_evaluator=dict(type='VOCMetric', metric='mAP', iou_thrs=[0.5, 0.75], eval_mode='11points'),
        test_evaluator=dict(type='VOCMetric', metric='mAP', iou_thrs=[0.5, 0.75], eval_mode='11points'),
        work_dir=f'{WORK_DIR_ROOT}/pascal_lod_mix_667x400{batch_suffix}_epoch{max_epochs}',
    )
    cfg['train_dataloader']['dataset'] = dict(
        type='RepeatDataset',
        times=PASCAL_MIX_REPEAT_TIMES,
        dataset=_voc_lod_mix_dataset('train', False),
    )
    cfg['val_dataloader']['dataset'] = _voc_lod_mix_dataset('val', True)
    cfg['test_dataloader']['dataset'] = _voc_lod_mix_dataset('val', True)
    return cfg


def build_lod_config():
    cfg = _common_runtime()
    cfg.update(
        experiment_config_name='RAWild_swint_lod',
        model=_model(len(LOD_CLASSES)),
        train_dataloader=_train_shell(),
        val_dataloader=_eval_shell(),
        test_dataloader=_eval_shell(),
        val_evaluator=dict(type='VOCMetric', metric='mAP', iou_thrs=[0.5, 0.75], eval_mode='11points'),
        test_evaluator=dict(type='VOCMetric', metric='mAP', iou_thrs=[0.5, 0.75], eval_mode='11points'),
        work_dir=f'{WORK_DIR_ROOT}/lod',
    )
    cfg['model']['backbone'].update(bit_depth_norm_max=16.0, bit_embed_min=8, bit_embed_max=16)
    cfg['train_dataloader']['dataset'] = dict(
        type='RepeatDataset',
        times=LOD_MIX_REPEAT_TIMES,
        dataset=_lod_dataset('train', False),
    )
    cfg['val_dataloader']['dataset'] = _lod_dataset('val', True)
    cfg['test_dataloader']['dataset'] = _lod_dataset('val', True)
    return cfg


def build_rod_config(
    resize_scale=COCO_RESIZE,
    train_repeat_times=1,
    max_epochs=MAX_EPOCHS,
    train_batch_size=1,
):
    tag = _scale_tag(resize_scale)
    repeat_suffix = '' if train_repeat_times == 1 else f'_repeat{train_repeat_times}'
    batch_suffix = '' if train_batch_size == 1 else f'_bs{train_batch_size}'
    cfg = _common_runtime(max_epochs=max_epochs)
    cfg.update(
        experiment_config_name=f'RAWild_swint_rod_{tag}{repeat_suffix}{batch_suffix}_epoch{max_epochs}',
        model=_model(len(ROD_CLASSES)),
        train_dataloader=_train_shell(batch_size=train_batch_size),
        val_dataloader=_eval_shell(),
        test_dataloader=_eval_shell(),
        val_evaluator=dict(
            type='CocoMetric',
            ann_file=f"{DATA_ROOTS['rod']}/annotations/val.json",
            metric='bbox',
            format_only=False,
            backend_args=BACKEND_ARGS,
        ),
        test_evaluator=dict(
            type='CocoMetric',
            ann_file=f"{DATA_ROOTS['rod']}/annotations/val.json",
            metric='bbox',
            format_only=False,
            backend_args=BACKEND_ARGS,
        ),
        work_dir=f'{WORK_DIR_ROOT}/rod_{tag}{repeat_suffix}{batch_suffix}_epoch{max_epochs}',
        load_from=CHECKPOINTS['rod_detector'],
    )
    cfg['model']['backbone'].update(pretrained=None, bit_depth_norm_max=16.0, bit_embed_min=8, bit_embed_max=16)
    cfg['model']['bbox_head']['num_classes'] = len(ROD_CLASSES)
    cfg['model']['test_cfg']['max_per_img'] = 10
    train_dataset = _rod_dataset('train', False, resize_scale)
    if train_repeat_times != 1:
        train_dataset = dict(
            type='RepeatDataset',
            times=train_repeat_times,
            dataset=train_dataset,
        )
    cfg['train_dataloader']['dataset'] = train_dataset
    cfg['val_dataloader']['dataset'] = _rod_dataset('val', True, resize_scale)
    cfg['test_dataloader']['dataset'] = _rod_dataset('val', True, resize_scale)
    return cfg


def build_aodraw_config(
    resize_scale=COCO_RESIZE,
    train_repeat_times=1,
    max_epochs=MAX_EPOCHS,
    train_batch_size=1,
):
    tag = _scale_tag(resize_scale)
    repeat_suffix = '' if train_repeat_times == 1 else f'_repeat{train_repeat_times}'
    batch_suffix = '' if train_batch_size == 1 else f'_bs{train_batch_size}'
    cfg = _common_runtime(max_epochs=max_epochs)
    cfg.update(
        experiment_config_name=f'RAWild_swint_aodraw_{tag}{repeat_suffix}{batch_suffix}_epoch{max_epochs}',
        model=_model(len(AODRAW_CLASSES)),
        train_dataloader=_train_shell(batch_size=train_batch_size),
        val_dataloader=_eval_shell(),
        test_dataloader=_eval_shell(),
        val_evaluator=dict(
            type='CocoMetric',
            ann_file=f"{DATA_ROOTS['aodraw']}/AODRaw/annotations/test_annotations_downsample_scale3_bbox_min_size32.json",
            metric='bbox',
            format_only=False,
            backend_args=BACKEND_ARGS,
        ),
        test_evaluator=dict(
            type='CocoMetric',
            ann_file=f"{DATA_ROOTS['aodraw']}/AODRaw/annotations/test_annotations_downsample_scale3_bbox_min_size32.json",
            metric='bbox',
            format_only=False,
            backend_args=BACKEND_ARGS,
        ),
        work_dir=f'{WORK_DIR_ROOT}/aodraw_{tag}{repeat_suffix}{batch_suffix}_epoch{max_epochs}',
        load_from=CHECKPOINTS['aodraw_detector'],
    )
    cfg['model']['backbone']['pretrained'] = None
    cfg['model']['bbox_head']['num_classes'] = len(AODRAW_CLASSES)
    cfg['model']['test_cfg']['max_per_img'] = 10
    cfg['optim_wrapper']['optimizer']['weight_decay'] = 0.0001
    train_dataset = _aodraw_dataset('train', False, resize_scale)
    if train_repeat_times != 1:
        train_dataset = dict(
            type='RepeatDataset',
            times=train_repeat_times,
            dataset=train_dataset,
        )
    cfg['train_dataloader']['dataset'] = train_dataset
    cfg['val_dataloader']['dataset'] = _aodraw_dataset('test', True, resize_scale)
    cfg['test_dataloader']['dataset'] = _aodraw_dataset('test', True, resize_scale)
    return cfg
