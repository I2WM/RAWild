import os

_base_ = ['./base.py']

train_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            img_path='images/training_raw_over_exp',
            seg_map_path='annotations/training')))

val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            img_path='images/validation_raw_over_exp',
            seg_map_path='annotations/validation')))

test_dataloader = val_dataloader

work_dir = os.path.join(
    os.environ.get('RAWILD_MMSEG_WORK_DIR_ROOT', 'work_dirs/rawild_mmseg'),
    'RAWild_mitb0', 'oe')
