import os
from copy import deepcopy

default_scope = 'mmdet'

DATA_ROOT = os.environ.get('RAWILD_DATA_ROOT', 'data/RAWild_datasets')
PASCAL_NPY_ROOT = os.environ.get(
    'RAWILD_PASCAL_NPY_ROOT', f'{DATA_ROOT}/PASCAL_RAW_npy')
PRETRAINED_ROOT = os.environ.get('RAWILD_PRETRAINED_ROOT', 'checkpoints/pretrained')
RUNTIME_ROOT = os.environ.get('RAWILD_WORK_DIR_ROOT', 'work_dirs')
WORK_DIR_ROOT = f'{RUNTIME_ROOT}/RAWild_resnet50'
CHECKPOINTS = dict(
    resnet50_backbone=f'{PRETRAINED_ROOT}/resnet50_backbone.pth',
    rod_detector=f'{PRETRAINED_ROOT}/retinanet_r50_fpn_1x_coco_rod5_fullinit.pth',
    aodraw_detector=f'{PRETRAINED_ROOT}/retinanet_r50_fpn_1x_coco_aod_detectorinit.pth',
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
PACK_META_KEYS = (
    'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'raw_bit_depth'
)
BACKEND_ARGS = None
MAX_EPOCHS = 20
LR_MILESTONES = [16]
PASCAL_MIX_REPEAT_TIMES = 1
LOD_MIX_REPEAT_TIMES = 8

custom_imports = dict(
    imports=[
        'mmdet.models.data_preprocessors.raw_data_preprocessor',
        'mmdet.datasets.pascal_raw',
        'mmdet.datasets.lod_bmvc21',
        'mmdet.datasets.aodraw',
        'mmdet.datasets.transforms.load_npy',
        'mmdet.datasets.transforms.load_png_dual',
        'mmdet.models.backbones.RAW_resnet_DIA',
    ],
    allow_failed_imports=False,
)


def _copy(value):
    return deepcopy(value)


def _metainfo(classes):
    return dict(classes=classes, palette=None)


def _scale_tag(scale):
    return f'{scale[0]}x{scale[1]}'


def _retina_head(num_classes):
    return dict(
        type='RetinaHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128],
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
    )


def _model(num_classes):
    return dict(
        type='RetinaNet',
        data_preprocessor=dict(
            type='RAWDataPreprocessor',
            bgr_to_rgb=False,
            pad_size_divisor=32,
        ),
        backbone=dict(
            type='RAW_BilateralGrid_ResNet_DIA',
            depth=50,
            in_channels=6,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            grid_depth=8,
            transformer_dim=128,
            num_heads=4,
            num_layers=2,
            mlp_ratio=4.0,
            grid_drop_rate=0.0,
            curve_n=6,
            use_bezier=True,
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
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint=CHECKPOINTS['resnet50_backbone']),
        ),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_input',
            num_outs=5,
        ),
        bbox_head=_retina_head(num_classes),
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
            ),
            sampler=dict(type='PseudoSampler'),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        test_cfg=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
        ),
    )


def _train_shell(batch_size=4, num_workers=4):
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


def _voc_train_pipeline(loader, resize_scale=VOC_RESIZE):
    return [
        loader,
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', scale=resize_scale, keep_ratio=True),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PackDetInputs', meta_keys=PACK_META_KEYS),
    ]


def _voc_test_pipeline(loader, resize_scale=VOC_RESIZE):
    return [
        loader,
        dict(type='Resize', scale=resize_scale, keep_ratio=True),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='PackDetInputs', meta_keys=PACK_META_KEYS),
    ]


def _coco_train_pipeline(loader, resize_scale=COCO_RESIZE):
    return [
        loader,
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', scale=resize_scale, keep_ratio=True),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PackDetInputs', meta_keys=PACK_META_KEYS),
    ]


def _coco_test_pipeline(loader, resize_scale=COCO_RESIZE):
    return [
        loader,
        dict(type='Resize', scale=resize_scale, keep_ratio=True),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='PackDetInputs', meta_keys=PACK_META_KEYS),
    ]


def _pascal_dataset(light_key, split, test_mode, classes=PASCAL_CLASSES,
                    resize_scale=VOC_RESIZE):
    img_subdir = {
        'pas_low': 'original/demosaic_low',
        'pas_nm': 'original/raw',
        'pas_oe': 'original/demosaic_oe',
    }[light_key]
    npy_subdir = {
        'pas_low': f"{DATA_ROOTS['pascal_npy']}/demosaic_low_12bit",
        'pas_nm': f"{DATA_ROOTS['pascal_npy']}/demosaic_normal_12bit",
        'pas_oe': f"{DATA_ROOTS['pascal_npy']}/demosaic_oe_12bit",
    }[light_key]
    loader = dict(type='LoadImageFromNpy', bits=12, img_subdir_npy=npy_subdir, dual_precision=True)
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
            _voc_test_pipeline(loader, resize_scale)
            if test_mode else _voc_train_pipeline(loader, resize_scale)
        ),
        backend_args=BACKEND_ARGS,
    )
    if not test_mode:
        dataset['filter_cfg'] = dict(filter_empty_gt=True, min_size=32, bbox_min_size=32)
    return dataset


def _lod_dataset(split, test_mode, classes=LOD_CLASSES,
                 resize_scale=VOC_RESIZE):
    loader = dict(type='LoadImageFromPngDual', bits=16)
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
            _voc_test_pipeline(loader, resize_scale)
            if test_mode else _voc_train_pipeline(loader, resize_scale)
        ),
        backend_args=BACKEND_ARGS,
    )
    if not test_mode:
        dataset['filter_cfg'] = dict(filter_empty_gt=True, min_size=32, bbox_min_size=32)
    return dataset


def _rod_dataset(split, test_mode, resize_scale=COCO_RESIZE):
    loader = dict(type='LoadImageFromNpy', bits=16, img_subdir_npy='', dual_precision=True)
    return dict(
        type='CocoDataset',
        data_root=DATA_ROOTS['rod'],
        ann_file=f'annotations/{split}.json',
        data_prefix=dict(img=f'images/{split}/'),
        metainfo=_metainfo(ROD_CLASSES),
        test_mode=test_mode,
        pipeline=(
            _coco_test_pipeline(loader, resize_scale)
            if test_mode else _coco_train_pipeline(loader, resize_scale)
        ),
        backend_args=BACKEND_ARGS,
        filter_cfg=None if test_mode else dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
    )


def _aodraw_dataset(split, test_mode, resize_scale=COCO_RESIZE):
    loader = dict(type='LoadImageFromNpy', bits=8, img_subdir_npy='', dual_precision=True)
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
            _coco_test_pipeline(loader, resize_scale)
            if test_mode else _coco_train_pipeline(loader, resize_scale)
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
            dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
            dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=LR_MILESTONES, gamma=0.1),
        ],
        train_cfg=dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1),
        val_cfg=dict(type='ValLoop'),
        test_cfg=dict(type='TestLoop'),
        auto_scale_lr=dict(base_batch_size=16, enable=False),
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
        experiment_config_name=f'RAWild_resnet50_{light_key}',
        model=_model(3),
        train_dataloader=_train_shell(),
        val_dataloader=_eval_shell(),
        test_dataloader=_eval_shell(),
        val_evaluator=dict(type='VOCMetric', metric='mAP', iou_thrs=[0.5, 0.75], eval_mode='11points'),
        test_evaluator=dict(type='VOCMetric', metric='mAP', iou_thrs=[0.5, 0.75], eval_mode='11points'),
        work_dir=f'{WORK_DIR_ROOT}/{light_key}',
    )
    cfg['model']['backbone'].update(bit_depth_norm_max=12.0)
    cfg['train_dataloader']['dataset'] = dict(type='RepeatDataset', times=5, dataset=_pascal_dataset(light_key, 'train', False))
    cfg['val_dataloader']['dataset'] = _pascal_dataset(light_key, 'val', True)
    cfg['test_dataloader']['dataset'] = _pascal_dataset(light_key, 'val', True)
    return cfg


def build_pascal_lod_mix_config():
    cfg = _common_runtime()
    cfg.update(
        experiment_config_name='RAWild_resnet50_pascal_lod_mix_667x400_epoch20',
        model=_model(len(VOC_LOD_MIX_CLASSES)),
        train_dataloader=_train_shell(),
        val_dataloader=_eval_shell(),
        test_dataloader=_eval_shell(),
        val_evaluator=dict(type='VOCMetric', metric='mAP', iou_thrs=[0.5, 0.75], eval_mode='11points'),
        test_evaluator=dict(type='VOCMetric', metric='mAP', iou_thrs=[0.5, 0.75], eval_mode='11points'),
        work_dir=f'{WORK_DIR_ROOT}/pascal_lod_mix_667x400_epoch20',
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
        experiment_config_name='RAWild_resnet50_lod',
        model=_model(8),
        train_dataloader=_train_shell(),
        val_dataloader=_eval_shell(),
        test_dataloader=_eval_shell(),
        val_evaluator=dict(type='VOCMetric', metric='mAP', iou_thrs=[0.5, 0.75], eval_mode='11points'),
        test_evaluator=dict(type='VOCMetric', metric='mAP', iou_thrs=[0.5, 0.75], eval_mode='11points'),
        work_dir=f'{WORK_DIR_ROOT}/lod',
    )
    cfg['model']['backbone'].update(bit_depth_norm_max=16.0, dia_activation='exp_tanh', dia_exp_alpha=1.0)
    cfg['train_dataloader']['dataset'] = dict(type='RepeatDataset', times=8, dataset=_lod_dataset('train', False))
    cfg['val_dataloader']['dataset'] = _lod_dataset('val', True)
    cfg['test_dataloader']['dataset'] = _lod_dataset('val', True)
    return cfg


def build_rod_config(resize_scale=COCO_RESIZE, train_repeat_times=1):
    tag = _scale_tag(resize_scale)
    repeat_suffix = '' if train_repeat_times == 1 else f'_repeat{train_repeat_times}'
    cfg = _common_runtime()
    cfg.update(
        experiment_config_name=f'RAWild_resnet50_rod_{tag}{repeat_suffix}_epoch20',
        model=_model(5),
        train_dataloader=_train_shell(),
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
        work_dir=f'{WORK_DIR_ROOT}/rod_{tag}{repeat_suffix}_epoch20',
        load_from=CHECKPOINTS['rod_detector'],
    )
    cfg['model']['backbone'].update(init_cfg=None, bit_depth_norm_max=16.0)
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


def build_aodraw_config(resize_scale=COCO_RESIZE, train_repeat_times=1):
    tag = _scale_tag(resize_scale)
    repeat_suffix = '' if train_repeat_times == 1 else f'_repeat{train_repeat_times}'
    cfg = _common_runtime()
    cfg.update(
        experiment_config_name=f'RAWild_resnet50_aodraw_{tag}{repeat_suffix}_epoch20',
        model=_model(len(AODRAW_CLASSES)),
        train_dataloader=_train_shell(),
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
        work_dir=f'{WORK_DIR_ROOT}/aodraw_{tag}{repeat_suffix}_epoch20',
        load_from=CHECKPOINTS['aodraw_detector'],
    )
    cfg['model']['backbone'].update(init_cfg=None)
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
