"""Swin backbone with DIA bilateral-grid RAW adapter for the mmdet2 Swin repo."""

import logging
from collections import OrderedDict

import torch
import torch.nn as nn

from mmcv_custom.checkpoint import load_state_dict
from mmdet.utils import get_root_logger
from ..builder import BACKBONES
from .rawild_adapter.bilateral_grid_dia import BilateralGridAdapter_DIA
from .swin_transformer import SwinTransformer


def _unwrap_backbone_state_dict(checkpoint):
    """Extract backbone-only weights from a detector or backbone checkpoint."""
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        checkpoint = checkpoint['model']

    if not isinstance(checkpoint, dict):
        raise TypeError('Unsupported checkpoint structure for DIA Swin init.')

    state_dict = OrderedDict()
    for key, value in checkpoint.items():
        if key.startswith('module.backbone.'):
            state_dict[key[len('module.backbone.'):]] = value
        elif key.startswith('backbone.'):
            state_dict[key[len('backbone.'):]] = value
        elif key.startswith(('neck.', 'bbox_head.', 'roi_head.', 'rpn_head.',
                             'module.neck.', 'module.bbox_head.',
                             'module.roi_head.', 'module.rpn_head.')):
            continue
        elif key.startswith('module.'):
            state_dict[key[len('module.'):]] = value
        else:
            state_dict[key] = value
    return state_dict


@BACKBONES.register_module(name='RAW_BilateralGrid_SwinTransformer_DIA')
@BACKBONES.register_module(name='RAWildSwinTransformer')
class RAW_BilateralGrid_SwinTransformer_DIA(SwinTransformer):
    """Apply DIA RAW-to-RGB mapping before a Swin backbone.

    Expected input is 6-channel dual-precision RAW:
      - channels 0:3 predict the bilateral grid
      - channels 3:6 apply the predicted grid
    """

    def __init__(self,
                 grid_depth=8,
                 transformer_dim=128,
                 num_heads=4,
                 num_layers=4,
                 mlp_ratio=4.0,
                 grid_drop_rate=0.0,
                 curve_n=None,
                 use_bezier=None,
                 use_grid=True,
                 bezier_type=None,
                 dia_k=0.2,
                 dia_activation='exp_tanh',
                 dia_exp_alpha=1.0,
                 dia_matrix_mode='d_ia',
                 cond_injection_type='none',
                 hist_bins=64,
                 hist_sigma=0.1,
                 hist_downsample_factor=16,
                 hist_prefix_tokens=8,
                 bit_embed_min=8,
                 bit_embed_max=24,
                 bit_depth_norm_max=24.0,
                 input_max_value=None,
                 pretrain_img_size=224,
                 in_chans=3,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 num_heads_swin=(3, 6, 12, 24),
                 window_size=7,
                 out_indices=(1, 2, 3),
                 drop_path_rate=0.2,
                 use_checkpoint=False,
                 frozen_stages=-1,
                 pretrained=None,
                 ape=False,
                 patch_norm=True):
        if in_chans != 3:
            raise ValueError(
                'RAW_BilateralGrid_SwinTransformer_DIA expects in_chans=3 '
                f'after DIA RGB mapping, but got in_chans={in_chans}.')

        self._pretrained = pretrained
        self.needs_img_metas = cond_injection_type != 'none'
        self.needs_data_samples = self.needs_img_metas

        super().__init__(
            pretrain_img_size=pretrain_img_size,
            patch_size=4,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=list(depths),
            num_heads=list(num_heads_swin),
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            norm_layer=nn.LayerNorm,
            ape=ape,
            patch_norm=patch_norm,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            use_checkpoint=use_checkpoint)

        self.bilateral_grid_adapter = BilateralGridAdapter_DIA(
            grid_depth=grid_depth,
            transformer_dim=transformer_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            drop_rate=grid_drop_rate,
            curve_n=curve_n,
            use_bezier=use_bezier,
            use_grid=use_grid,
            bezier_type=bezier_type,
            dia_k=dia_k,
            dia_activation=dia_activation,
            dia_exp_alpha=dia_exp_alpha,
            dia_matrix_mode=dia_matrix_mode,
            cond_injection_type=cond_injection_type,
            hist_bins=hist_bins,
            hist_sigma=hist_sigma,
            hist_downsample_factor=hist_downsample_factor,
            hist_prefix_tokens=hist_prefix_tokens,
            bit_embed_min=bit_embed_min,
            bit_embed_max=bit_embed_max,
            bit_depth_norm_max=bit_depth_norm_max,
        )

    def init_weights(self, pretrained=None):
        """Initialize backbone, supporting detector checkpoints as source."""
        logger = get_root_logger()
        pretrained = pretrained or self._pretrained

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)
                if getattr(m, 'weight', None) is not None:
                    nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)
        if pretrained is None:
            return
        if not isinstance(pretrained, str):
            raise TypeError('pretrained must be a str or None')

        checkpoint = torch.load(pretrained, map_location='cpu')
        state_dict = _unwrap_backbone_state_dict(checkpoint)
        load_state_dict(self, state_dict, strict=False, logger=logger)
        logging.getLogger(__name__).info(
            'Loaded DIA Swin backbone init from detector checkpoint: %s',
            pretrained)

    @staticmethod
    def _extract_metainfo(sample):
        if hasattr(sample, 'metainfo'):
            return sample.metainfo
        if hasattr(sample, 'metainfo_items'):
            return dict(sample.metainfo_items())
        return sample

    @staticmethod
    def _collect_raw_bit_depth(img_metas, device, dtype):
        if img_metas is None:
            raise ValueError(
                'DIA Swin with conditional injection requires img_metas.')
        values = []
        for idx, meta in enumerate(img_metas):
            value = meta.get('raw_bit_depth')
            if value is None:
                raise KeyError(
                    f'Missing raw_bit_depth in img_metas for sample index {idx}.')
            values.append(float(value))
        return torch.tensor(values, device=device, dtype=dtype).view(-1, 1)

    def forward(self, x, img_metas=None, batch_data_samples=None):
        x = x.float()
        if x.shape[1] != 6:
            raise ValueError(
                'RAW_BilateralGrid_SwinTransformer_DIA expects 6-channel '
                f'input, got {x.shape[1]} channels.')

        if img_metas is None and batch_data_samples is not None:
            img_metas = [self._extract_metainfo(sample) for sample in batch_data_samples]

        raw_bit_depth = None
        if self.needs_img_metas:
            raw_bit_depth = self._collect_raw_bit_depth(
                img_metas, device=x.device, dtype=x.dtype)

        x_rgb = self.bilateral_grid_adapter(
            x[:, :3, :, :], x[:, 3:, :, :], raw_bit_depth=raw_bit_depth)
        return tuple(super().forward(x_rgb))
