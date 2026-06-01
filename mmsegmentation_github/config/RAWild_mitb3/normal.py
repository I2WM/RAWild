import os

_base_ = ['./base.py']

work_dir = os.path.join(
    os.environ.get('RAWILD_MMSEG_WORK_DIR_ROOT', 'work_dirs/rawild_mmseg'),
    'RAWild_mitb3', 'normal')
