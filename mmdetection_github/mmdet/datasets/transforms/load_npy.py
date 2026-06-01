# Copyright (c) OpenMMLab. All rights reserved.
"""Load NPY images and normalize them with an explicit bit depth."""

import os
from typing import Optional

import numpy as np
from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS
from mmdet.utils.rawild_debug_trace import maybe_dump_nonfinite


def _set_default_image_meta(results: dict, img: np.ndarray, img_path: str) -> None:
    results['img'] = img
    results['img_shape'] = img.shape[:2]
    results['ori_shape'] = img.shape[:2]
    results['img_path'] = img_path
    results.setdefault('scale_factor', (1.0, 1.0))
    results.setdefault('flip', False)
    results.setdefault('flip_direction', None)


@TRANSFORMS.register_module()
class LoadImageFromNpy(BaseTransform):
    """Load an NPY image and optionally duplicate it to 6 channels.

    Args:
        bits (int): Bit depth used for normalization.
        img_subdir_npy (str): If set, replace the image directory name in
            ``img_path`` with this subdirectory.
        dual_precision (bool): Whether to concatenate the normalized image with
            itself along the channel dimension.
    """

    def __init__(self,
                 bits,
                 img_subdir_npy='',
                 dual_precision=False,
                 clip_before_normalize=False,
                 clip_after_normalize=False,
                 path_key_from: Optional[str] = None):
        self.bits = int(bits)
        self.img_subdir_npy = img_subdir_npy
        self.dual_precision = bool(dual_precision)
        self.clip_before_normalize = bool(clip_before_normalize)
        self.clip_after_normalize = bool(clip_after_normalize)
        self.path_key_from = path_key_from

    def _resolve_npy_path(self, results: dict) -> str:
        img_path = results['img_path']

        if self.img_subdir_npy:
            if self.path_key_from:
                image_key = results.get(self.path_key_from)
                if image_key is None:
                    raise KeyError(
                        f'Results does not contain path key {self.path_key_from!r} '
                        f'needed by {self.__class__.__name__}.')
                rel_parts = str(image_key).replace('\\', '/').strip('/').split('/')
                return os.path.join(self.img_subdir_npy, *rel_parts) + '.npy'

            dirname = os.path.dirname(img_path)
            parent = os.path.dirname(dirname)
            basename = os.path.basename(img_path)
            stem, _ = os.path.splitext(basename)
            return os.path.join(parent, self.img_subdir_npy, stem + '.npy')

        if not img_path.endswith('.npy'):
            stem, _ = os.path.splitext(img_path)
            return stem + '.npy'
        return img_path

    def transform(self, results: dict) -> dict:
        img_path = self._resolve_npy_path(results)

        img = np.load(img_path)
        if img is None:
            raise FileNotFoundError(f'Failed to load image: {img_path}')
        if img.ndim != 3:
            raise ValueError(f'Expected a 3D NPY array, got shape={img.shape} for {img_path}')
        if img.shape[2] != 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        if img.shape[2] != 3:
            raise ValueError(f'Expected 3 channels, got shape={img.shape} for {img_path}')

        denom = float(max((2**self.bits) - 1, 1))
        img = img.astype(np.float32)
        if self.clip_before_normalize:
            img = np.clip(img, 0.0, denom)
        img = img / denom
        if self.clip_after_normalize:
            img = np.clip(img, 0.0, 1.0)

        if self.dual_precision:
            img = np.concatenate([img, img], axis=-1)

        _set_default_image_meta(results, img, img_path)
        results['raw_bit_depth'] = self.bits
        maybe_dump_nonfinite(
            'loader.load_npy',
            {'img': img},
            extra=dict(
                img_path=img_path,
                bits=self.bits,
                dual_precision=self.dual_precision,
                clip_before_normalize=self.clip_before_normalize,
                clip_after_normalize=self.clip_after_normalize,
                img_shape=list(img.shape)),
            raise_on_nonfinite=True)
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'bits={self.bits!r}, '
                f'img_subdir_npy={self.img_subdir_npy!r}, '
                f'dual_precision={self.dual_precision!r}, '
                f'clip_before_normalize={self.clip_before_normalize!r}, '
                f'clip_after_normalize={self.clip_after_normalize!r}, '
                f'path_key_from={self.path_key_from!r})')
