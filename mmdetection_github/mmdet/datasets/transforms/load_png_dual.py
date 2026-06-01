# Copyright (c) OpenMMLab. All rights reserved.
"""自定义 Transform: 加载 PNG 并输出 bilateral grid 兼容格式。

LOD 数据集的 RAW_dark 图像是 16-bit RGB PNG。
历史上 RAW_BilateralGrid_ResNet 使用 6 通道输入。
这里保留 6 通道输出，并统一按显式 bits 归一化到 [0, 1]。
"""
import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImageFromPngDual(BaseTransform):
    """Load a PNG image and produce 6-channel output (legacy-compatible).

    Args:
        bits (int): Bit depth of the stored PNG values.
        img_subdir (str): If set, replace the image directory name in img_path
            with this subdirectory. Default: '' (use original path).
    """

    def __init__(self, bits, img_subdir=''):
        self.img_subdir = img_subdir
        self.bits = bits

    def transform(self, results: dict) -> dict:
        img_path = results['img_path']

        # Optionally redirect to a different subdirectory
        if self.img_subdir:
            import os
            dirname = os.path.dirname(img_path)
            parent = os.path.dirname(dirname)
            basename = os.path.basename(img_path)
            img_path = os.path.join(parent, self.img_subdir, basename)

        # Load PNG (preserve bit-depth), then BGR → RGB
        # Note: IMREAD_UNCHANGED keeps uint8/uint16 as-is.
        img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            raise FileNotFoundError(f'Failed to load image: {img_path}')
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError(
                f'Expected 3-channel PNG, got shape={img_bgr.shape} for {img_path}')
        if img_bgr.dtype == np.uint16 and int(self.bits) <= 8:
            actual_max = int(img_bgr.max())
            if actual_max > 255:
                raise ValueError(
                    'Bit-depth mismatch for '
                    f'{img_path}: image dtype={img_bgr.dtype}, max={actual_max}, '
                    f'but LoadImageFromPngDual(bits={self.bits}) was configured. '
                    'Use bits=16 for RAW_dark PNG inputs.')
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 严格按显式 bits 做归一化，不做自动兼容分支
        denom = float((2**int(self.bits)) - 1)
        denom = max(denom, 1.0)
        img_normalized = img_rgb.astype(np.float32) / denom

        # 两组通道都输出 [0, 1] 范围
        # Concatenate to 6 channels: [H, W, 6]
        img = np.concatenate([img_normalized, img_normalized], axis=-1)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        results['img_path'] = img_path
        results['raw_bit_depth'] = int(self.bits)

        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'img_subdir={self.img_subdir!r}, '
                f'bits={self.bits!r})')
