import warnings

import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.registry import MODELS
from ..layers import ResLayer
from .rawild_adapter.bilateral_grid_dia import BilateralGridAdapter_DIA
from .resnet import BasicBlock, Bottleneck


@MODELS.register_module()
class RAW_BilateralGrid_ResNet_DIA(BaseModule):
    """ResNet backbone with a DIA bilateral-grid adapter."""

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
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
                 hist_condition_mode='quantile',
                 hist_prefix_tokens=8,
                 bit_embed_min=8,
                 bit_embed_max=24,
                 bit_depth_norm_max=24.0,
                 input_max_value=None,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(type='Constant', val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type='Constant', val=0,
                            override=dict(name='norm2'))
                    elif block is Bottleneck:
                        block_init_cfg = dict(
                            type='Constant', val=0,
                            override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.strides = strides
        self.dilations = dilations
        self.out_indices = out_indices
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        self.plugins = plugins
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels
        self.needs_data_samples = cond_injection_type != 'none'

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
            hist_condition_mode=hist_condition_mode,
            hist_prefix_tokens=hist_prefix_tokens,
            bit_embed_min=bit_embed_min,
            bit_embed_max=bit_embed_max,
            bit_depth_norm_max=bit_depth_norm_max,
        )

        self._make_stem_layer(3, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            stage_dcn = self.dcn if self.stage_with_dcn[i] else None
            stage_plugins = self.make_stage_plugins(plugins, i) \
                if plugins is not None else None
            planes = base_channels * 2 ** i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=stage_dcn,
                plugins=stage_plugins,
                init_cfg=block_init_cfg)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()
        self.feat_dim = self.block.expansion * base_channels * 2 ** (
            len(self.stage_blocks) - 1)

    def make_stage_plugins(self, plugins, stage_idx):
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)
        return stage_plugins

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(self.conv_cfg, in_channels,
                                 stem_channels // 2, kernel_size=3,
                                 stride=2, padding=1, bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(self.conv_cfg, stem_channels // 2,
                                 stem_channels // 2, kernel_size=3,
                                 stride=1, padding=1, bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(self.conv_cfg, stem_channels // 2,
                                 stem_channels, kernel_size=3,
                                 stride=1, padding=1, bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg, in_channels, stem_channels,
                kernel_size=7, stride=2, padding=3, bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for module in [self.conv1, self.norm1]:
                    for param in module.parameters():
                        param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            module = getattr(self, f'layer{i}')
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    @staticmethod
    def _extract_metainfo(sample):
        if hasattr(sample, 'metainfo'):
            return sample.metainfo
        if hasattr(sample, 'metainfo_items'):
            return dict(sample.metainfo_items())
        return {}

    def _collect_raw_bit_depth(self, batch_data_samples, device, dtype):
        if batch_data_samples is None:
            raise ValueError(
                'RAW_BilateralGrid_ResNet_DIA with conditional injection '
                'requires batch_data_samples to be passed from the detector.')

        values = []
        for idx, sample in enumerate(batch_data_samples):
            meta = self._extract_metainfo(sample)
            value = meta.get('raw_bit_depth')
            if value is None:
                raise KeyError(
                    'Missing raw_bit_depth in batch_data_samples '
                    f'for sample index {idx}.')
            values.append(float(value))
        return torch.tensor(values, device=device, dtype=dtype).view(-1, 1)

    def forward(self, x, batch_data_samples=None):
        x = x.float()
        if x.shape[1] != 6:
            raise ValueError(
                'RAW_BilateralGrid_ResNet_DIA expects 6-channel input, '
                f'got {x.shape[1]} channels.')

        x_guide = x[:, :3, :, :]
        x_apply = x[:, 3:, :, :]
        raw_bit_depth = None
        if self.needs_data_samples:
            raw_bit_depth = self._collect_raw_bit_depth(
                batch_data_samples, device=x.device, dtype=x.dtype)

        x_rgb = self.bilateral_grid_adapter(
            x_guide, x_apply, raw_bit_depth=raw_bit_depth)

        x = x_rgb
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for name, module in self.named_modules():
                # Keep BN layers inside the bilateral-grid adapter trainable.
                if (isinstance(module, _BatchNorm) and
                        not name.startswith('bilateral_grid_adapter.')):
                    module.eval()

