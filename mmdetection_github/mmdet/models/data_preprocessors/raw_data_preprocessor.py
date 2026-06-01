from typing import Optional, Sequence, Union

import torch

from mmdet.registry import MODELS
from .data_preprocessor import DetDataPreprocessor

@MODELS.register_module()
class RAWDataPreprocessor(DetDataPreprocessor):
    """Image pre-processor for RAW data.

    由于 mmengine 的 `ImgDataPreprocessor.__init__` 会限制 mean/std 只能是 1 或 3 个值，
    这里先以 mean=None/std=None 初始化以绕过该断言，然后手动注册任意通道数的 mean/std buffer。

    注意：一旦注册了 mean/std，本类仍复用父类 forward（内部执行 (x-mean)/std），
    因此 mean/std 必须是 Tensor，形状为 [C, 1, 1] 才能正确广播到 [C, H, W]。
    """

    def __init__(self,
                 mean: Optional[Sequence[Union[float, int]]] = None,
                 std: Optional[Sequence[Union[float, int]]] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 mask_pad_value: int = 0,
                 seg_pad_value: int = 255,
                 boxtype2tensor: bool = True,
                 non_blocking: Optional[bool] = False,
                 batch_augments=None):
        super().__init__(
            mean=None,  # Pass None initially to bypass assert
            std=None,   # Pass None initially to bypass assert
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            mask_pad_value=mask_pad_value,
            seg_pad_value=seg_pad_value,
            boxtype2tensor=boxtype2tensor,
            non_blocking=non_blocking,
            batch_augments=batch_augments)

        # Manually register mean/std for arbitrary channels if provided
        if (mean is None) != (std is None):
            raise ValueError('mean and std should be both None or both provided')

        if mean is not None:
            mean_t = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
            std_t = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
            if mean_t.numel() != std_t.numel():
                raise ValueError(
                    f'len(mean)={mean_t.numel()} must match len(std)={std_t.numel()}')

            # Enable normalization in parent forward
            self._enable_normalize = True
            self.register_buffer('mean', mean_t, persistent=False)
            self.register_buffer('std', std_t, persistent=False)
