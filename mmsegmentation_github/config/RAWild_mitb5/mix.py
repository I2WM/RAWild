import os

_base_ = ['./base.py']

dataset_type = 'ADE20KDataset'
data_root = os.environ.get('RAWILD_MMSEG_DATA_ROOT', 'data/ADE20K/ADEChallengeData2016')
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                data_prefix=dict(
                    img_path='images/training_raw_low',
                    seg_map_path='annotations/training'),
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                data_root=data_root,
                data_prefix=dict(
                    img_path='images/training_raw_over_exp',
                    seg_map_path='annotations/training'),
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                data_root=data_root,
                data_prefix=dict(
                    img_path='images/training_raw',
                    seg_map_path='annotations/training'),
                pipeline=train_pipeline),
        ]))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation_raw_low',
            seg_map_path='annotations/validation')))

test_dataloader = val_dataloader

work_dir = os.path.join(
    os.environ.get('RAWILD_MMSEG_WORK_DIR_ROOT', 'work_dirs/rawild_mmseg'),
    'RAWild_mitb5', 'mix')
