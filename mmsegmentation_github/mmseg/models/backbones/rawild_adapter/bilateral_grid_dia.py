import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule

from .rgbuv_histogram import RGBuvHistogram

_logger = logging.getLogger(__name__)


def _fmt(arr, precision=4):
    import numpy as np
    return '[' + ', '.join(f'{v:.{precision}f}' for v in arr.flatten()) + ']'


def _log_info(msg):
    if _logger.isEnabledFor(logging.INFO):
        _logger.info(msg)


class ConditionalTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0,
                 cond_injection_type='none'):
        super().__init__()
        self.cond_injection_type = cond_injection_type
        self.cond_token_dim = dim * 2
        self.bit_cond_dim = dim

        self.norm1 = nn.LayerNorm(
            dim, elementwise_affine=(
                cond_injection_type not in {
                    'adaln', 'hist_prefix_bit_adaln'
                }))
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

        self.cross_norm = None
        self.cond_norm = None
        self.cross_attn = None
        self.adaln_proj = None
        self.film_proj = None
        self.bias_proj = None

        if cond_injection_type == 'cross_attn':
            self.cross_norm = nn.LayerNorm(dim)
            self.cond_norm = nn.LayerNorm(dim)
            self.cross_attn = nn.MultiheadAttention(
                dim, num_heads, dropout=drop, batch_first=True)
            nn.init.zeros_(self.cross_attn.out_proj.weight)
            nn.init.zeros_(self.cross_attn.out_proj.bias)
        elif cond_injection_type in {'adaln', 'hist_prefix_bit_adaln'}:
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
            adaln_in_dim = (
                self.cond_token_dim if cond_injection_type == 'adaln'
                else self.bit_cond_dim)
            self.adaln_proj = nn.Linear(adaln_in_dim, dim * 6)
            nn.init.zeros_(self.adaln_proj.weight)
            nn.init.zeros_(self.adaln_proj.bias)
        elif cond_injection_type == 'film':
            self.film_proj = nn.Linear(self.cond_token_dim, dim * 2)
            nn.init.zeros_(self.film_proj.weight)
            nn.init.zeros_(self.film_proj.bias)
        elif cond_injection_type == 'add_bias':
            self.bias_proj = nn.Linear(self.cond_token_dim, dim)
            nn.init.zeros_(self.bias_proj.weight)
            nn.init.zeros_(self.bias_proj.bias)

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def _require_cond_tokens(self, cond_tokens):
        if cond_tokens is None:
            raise ValueError(
                f'{self.cond_injection_type} requires cond_tokens, got None.')

    def _flatten_cond_tokens(self, cond_tokens):
        self._require_cond_tokens(cond_tokens)
        return cond_tokens.flatten(1)

    def forward(self, x, cond_tokens=None, bit_cond=None):
        if self.cond_injection_type == 'none':
            x2 = self.norm1(x)
            x = x + self.attn(x2, x2, x2, need_weights=False)[0]
            x = x + self.mlp(self.norm2(x))
            return x

        if self.cond_injection_type == 'adaln':
            # x: [B, N, D]
            # cond_tokens: [B, 2, D] = [bit_token, hist_token]
            cond_input = self._flatten_cond_tokens(cond_tokens)
            # cond_input: [B, 2D]
            shift1, scale1, gate_attn, shift2, scale2, gate_mlp = self.adaln_proj(cond_input).chunk(6, dim=-1)
            # shift*/scale*/gate*: [B, D]
            x1 = self._modulate(self.norm1(x), shift1, scale1)
            # norm1(x): [B, N, D], x1: [B, N, D]
            attn_out = self.attn(x1, x1, x1, need_weights=False)[0]
            # attn_out: [B, N, D], tanh(gate_attn).unsqueeze(1): [B, 1, D]
            x = x + torch.tanh(gate_attn).unsqueeze(1) * attn_out
            x2 = self._modulate(self.norm2(x), shift2, scale2)
            # norm2(x): [B, N, D], x2: [B, N, D]
            mlp_out = self.mlp(x2)
            # mlp_out: [B, N, D], tanh(gate_mlp).unsqueeze(1): [B, 1, D]
            x = x + torch.tanh(gate_mlp).unsqueeze(1) * mlp_out
            # return x: [B, N, D]
            return x

        if self.cond_injection_type == 'hist_prefix_bit_adaln':
            # x: [B, N_joint, D]
            # bit_cond: [B, D]
            if bit_cond is None:
                raise ValueError(
                    'hist_prefix_bit_adaln requires bit_cond, got None.')
            shift1, scale1, gate_attn, shift2, scale2, gate_mlp = (
                self.adaln_proj(bit_cond).chunk(6, dim=-1))
            # shift*/scale*/gate*: [B, D]
            x1 = self._modulate(self.norm1(x), shift1, scale1)
            # norm1(x): [B, N_joint, D], x1: [B, N_joint, D]
            attn_out = self.attn(x1, x1, x1, need_weights=False)[0]
            # attn_out: [B, N_joint, D], tanh(gate_attn).unsqueeze(1): [B, 1, D]
            x = x + torch.tanh(gate_attn).unsqueeze(1) * attn_out
            x2 = self._modulate(self.norm2(x), shift2, scale2)
            # norm2(x): [B, N_joint, D], x2: [B, N_joint, D]
            mlp_out = self.mlp(x2)
            # mlp_out: [B, N_joint, D], tanh(gate_mlp).unsqueeze(1): [B, 1, D]
            x = x + torch.tanh(gate_mlp).unsqueeze(1) * mlp_out
            # return x: [B, N_joint, D]
            return x

        if self.cond_injection_type == 'cross_attn':
            self._require_cond_tokens(cond_tokens)
            # x: [B, N, D], cond_tokens: [B, 2, D]
            x1 = self.norm1(x)
            x = x + self.attn(x1, x1, x1, need_weights=False)[0]
            q = self.cross_norm(x)
            kv = self.cond_norm(cond_tokens)
            # q: [B, N, D], kv: [B, 2, D]
            x = x + self.cross_attn(q, kv, kv, need_weights=False)[0]
            x = x + self.mlp(self.norm2(x))
            # return x: [B, N, D]
            return x

        if self.cond_injection_type == 'prefix':
            self._require_cond_tokens(cond_tokens)
            # x: [B, N, D], cond_tokens: [B, 2, D]
            joint = torch.cat([cond_tokens, x], dim=1)
            # joint: [B, N + 2, D]
            joint2 = self.norm1(joint)
            joint = joint + self.attn(joint2, joint2, joint2,
                                      need_weights=False)[0]
            joint = joint + self.mlp(self.norm2(joint))
            # drop the 2 prefix tokens and keep the original main sequence
            return joint[:, cond_tokens.shape[1]:]

        if self.cond_injection_type == 'film':
            # x: [B, N, D], cond_tokens: [B, 2, D]
            cond_input = self._flatten_cond_tokens(cond_tokens)
            # cond_input: [B, 2D]
            x1 = self.norm1(x)
            x = x + self.attn(x1, x1, x1, need_weights=False)[0]
            gamma, beta = self.film_proj(cond_input).chunk(2, dim=-1)
            # gamma/beta: [B, D] -> unsqueeze(1): [B, 1, D]
            x = x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
            x = x + self.mlp(self.norm2(x))
            # return x: [B, N, D]
            return x

        if self.cond_injection_type == 'add_bias':
            # x: [B, N, D], cond_tokens: [B, 2, D]
            cond_input = self._flatten_cond_tokens(cond_tokens)
            # cond_input: [B, 2D]
            bias = self.bias_proj(cond_input).unsqueeze(1)
            # bias: [B, 1, D], broadcast to every token in x
            x1 = self.norm1(x + bias)
            x = x + self.attn(x1, x1, x1, need_weights=False)[0]
            x = x + self.mlp(self.norm2(x))
            # return x: [B, N, D]
            return x

        raise ValueError(f'Unsupported cond_injection_type: '
                         f'{self.cond_injection_type}')


class BilateralGridAdapter_DIA(BaseModule):
    """DIA bilateral-grid adapter with optional bit+hist conditioning."""

    def __init__(
        self,
        grid_depth=8,
        transformer_dim=128,
        num_heads=4,
        num_layers=2,
        mlp_ratio=4.0,
        drop_rate=0.0,
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
        hist_condition_mode='quantile',
    ):
        super().__init__()

        required = {
            'curve_n': curve_n,
            'use_bezier': use_bezier,
            'bezier_type': bezier_type,
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            raise ValueError(
                'BilateralGridAdapter_DIA requires explicit config values for '
                f'{missing}.')

        valid_cond_types = {
            'none', 'adaln', 'cross_attn', 'prefix', 'film', 'add_bias',
            'hist_prefix_bit_adaln'
        }
        if cond_injection_type not in valid_cond_types:
            raise ValueError(
                f'Invalid cond_injection_type={cond_injection_type!r}. '
                f'Expected one of {sorted(valid_cond_types)}.')

        self.grid_depth = grid_depth
        self.transformer_dim = transformer_dim
        self.use_bezier = use_bezier
        self.use_grid = use_grid
        self.bezier_type = bezier_type
        self.curve_n = curve_n
        self.dia_k = dia_k
        self.dia_activation = dia_activation
        self.dia_exp_alpha = dia_exp_alpha
        self.dia_matrix_mode = dia_matrix_mode
        self.cond_injection_type = cond_injection_type
        self.bit_depth_norm_max = float(bit_depth_norm_max)
        self.use_cond_injection = cond_injection_type != 'none'
        self.hist_bins = int(hist_bins)
        self.hist_sigma = float(hist_sigma)
        self.hist_downsample_factor = int(hist_downsample_factor)
        self.hist_prefix_tokens = int(hist_prefix_tokens)
        self.bit_embed_min = int(bit_embed_min)
        self.bit_embed_max = int(bit_embed_max)
        self.hist_condition_mode = hist_condition_mode
        self.use_hist_prefix_bit_adaln = (
            cond_injection_type == 'hist_prefix_bit_adaln')

        if self.hist_bins <= 0:
            raise ValueError('hist_bins must be > 0.')
        if self.hist_sigma <= 0:
            raise ValueError('hist_sigma must be > 0.')
        if self.hist_downsample_factor <= 0:
            raise ValueError('hist_downsample_factor must be > 0.')
        if self.hist_prefix_tokens <= 0:
            raise ValueError('hist_prefix_tokens must be > 0.')
        if self.bit_embed_min > self.bit_embed_max:
            raise ValueError('bit_embed_min must be <= bit_embed_max.')
        valid_hist_modes = {'quantile', 'legacy_conv'}
        if self.hist_condition_mode not in valid_hist_modes:
            raise ValueError(
                f'Invalid hist_condition_mode={self.hist_condition_mode!r}. '
                f'Expected one of {sorted(valid_hist_modes)}.')

        valid_dia_modes = {'d_ia', 'd_only', 'ia_only', 'full_affine'}
        if self.dia_matrix_mode not in valid_dia_modes:
            raise ValueError(
                f'Invalid dia_matrix_mode={self.dia_matrix_mode!r}. '
                f'Expected one of {sorted(valid_dia_modes)}.')

        self.C_grid = 9
        if self.use_bezier:
            self.C_curve = self.curve_n - 1 if self.bezier_type == 'init_bias' \
                else self.curve_n
        else:
            self.C_curve = 0
        self.C_total = self.C_grid
        self.register_buffer(
            'hist_quantiles',
            torch.arange(1, self.hist_bins + 1, dtype=torch.float32) /
            float(self.hist_bins),
            persistent=False)
        self.register_buffer(
            'hist_value_grid',
            torch.linspace(0.0, 1.0, steps=self.hist_bins,
                           dtype=torch.float32),
            persistent=False)

        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, transformer_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(transformer_dim),
            nn.GELU(),
        )

        self.transformer_blocks = nn.ModuleList([
            ConditionalTransformerBlock(
                transformer_dim,
                num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                cond_injection_type=cond_injection_type,
            )
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(transformer_dim)

        n_coeffs = self.grid_depth * self.C_grid
        self.grid_head = nn.Conv2d(transformer_dim, n_coeffs, 1, bias=True)

        if self.use_bezier:
            self.curve_head = nn.Linear(transformer_dim, 3 * self.C_curve)
        else:
            self.curve_head = None

        self.histogram = None
        self.hist_encoder = None
        self.hist_token_encoder = None
        self.bit_encoder = None
        if self.use_cond_injection:
            if self.use_hist_prefix_bit_adaln:
                self.histogram = RGBuvHistogram(
                    num_bins=self.hist_bins,
                    sigma=self.hist_sigma,
                    downsample_factor=self.hist_downsample_factor,
                )
                self.hist_encoder = nn.Sequential(
                    nn.Conv2d(1, transformer_dim, kernel_size=3, padding=1,
                              bias=False),
                    nn.BatchNorm2d(transformer_dim),
                    nn.GELU(),
                    nn.AdaptiveAvgPool2d((self.hist_prefix_tokens, 1)),
                )
                self.bit_encoder = nn.Embedding(
                    num_embeddings=self.bit_embed_max - self.bit_embed_min + 1,
                    embedding_dim=transformer_dim,
                )
            else:
                if self.hist_condition_mode == 'legacy_conv':
                    self.histogram = RGBuvHistogram(
                        num_bins=self.hist_bins,
                        sigma=self.hist_sigma,
                        downsample_factor=self.hist_downsample_factor,
                    )
                    self.hist_encoder = nn.Sequential(
                        nn.Conv2d(1, transformer_dim, kernel_size=3, padding=1,
                                  bias=False),
                        nn.BatchNorm2d(transformer_dim),
                        nn.GELU(),
                        nn.AdaptiveAvgPool2d(1),
                    )
                else:
                    quantile_feat_dim = 3 * int(self.hist_quantiles.numel())
                    self.hist_token_encoder = nn.Sequential(
                        nn.Linear(quantile_feat_dim, transformer_dim),
                        nn.GELU(),
                        nn.Linear(transformer_dim, transformer_dim),
                    )
                self.bit_encoder = nn.Sequential(
                    nn.Linear(1, transformer_dim),
                    nn.GELU(),
                    nn.Linear(transformer_dim, transformer_dim),
                )

        self._init_weights()
        self._fwd_count = 0

    def _init_weights(self):
        nn.init.zeros_(self.grid_head.weight)
        nn.init.zeros_(self.grid_head.bias)

        if self.use_bezier:
            nn.init.zeros_(self.curve_head.weight)
            if self.bezier_type == 'init_bias':
                nn.init.zeros_(self.curve_head.bias)
            else:
                nn.init.ones_(self.curve_head.bias)

    def _reconstruct_DIA(self, raw_grid):
        bsz, _, gd, gh, gw = raw_grid.shape
        device = raw_grid.device

        d_hat = raw_grid[:, 0:3]
        a_hat = raw_grid[:, 3:9]

        if self.dia_activation == 'exp_tanh':
            d = torch.exp(self.dia_exp_alpha * torch.tanh(d_hat))
        else:
            d = F.softplus(d_hat)

        a = self.dia_k * torch.tanh(a_hat)

        if self.dia_matrix_mode == 'd_ia':
            ip_a = torch.zeros(bsz, 3, 3, gd, gh, gw,
                               device=device, dtype=d_hat.dtype)
            ip_a[:, 0, 0] = 1.0
            ip_a[:, 1, 1] = 1.0
            ip_a[:, 2, 2] = 1.0
            ip_a[:, 0, 1] = a[:, 0]
            ip_a[:, 0, 2] = a[:, 1]
            ip_a[:, 1, 0] = a[:, 2]
            ip_a[:, 1, 2] = a[:, 3]
            ip_a[:, 2, 0] = a[:, 4]
            ip_a[:, 2, 1] = a[:, 5]
            matrix = d.unsqueeze(2) * ip_a
            return matrix.reshape(bsz, 9, gd, gh, gw)

        matrix = torch.zeros(bsz, 3, 3, gd, gh, gw,
                             device=device, dtype=d_hat.dtype)

        if self.dia_matrix_mode == 'd_only':
            matrix[:, 0, 0] = d[:, 0]
            matrix[:, 1, 1] = d[:, 1]
            matrix[:, 2, 2] = d[:, 2]
        elif self.dia_matrix_mode == 'ia_only':
            matrix[:, 0, 0] = 1.0
            matrix[:, 1, 1] = 1.0
            matrix[:, 2, 2] = 1.0
            matrix[:, 0, 1] = a[:, 0]
            matrix[:, 0, 2] = a[:, 1]
            matrix[:, 1, 0] = a[:, 2]
            matrix[:, 1, 2] = a[:, 3]
            matrix[:, 2, 0] = a[:, 4]
            matrix[:, 2, 1] = a[:, 5]
        elif self.dia_matrix_mode == 'full_affine':
            matrix[:, 0, 0] = d[:, 0]
            matrix[:, 1, 1] = d[:, 1]
            matrix[:, 2, 2] = d[:, 2]
            matrix[:, 0, 1] = a[:, 0]
            matrix[:, 0, 2] = a[:, 1]
            matrix[:, 1, 0] = a[:, 2]
            matrix[:, 1, 2] = a[:, 3]
            matrix[:, 2, 0] = a[:, 4]
            matrix[:, 2, 1] = a[:, 5]
        else:
            raise ValueError(f'Unsupported dia_matrix_mode: {self.dia_matrix_mode}')

        return matrix.reshape(bsz, 9, gd, gh, gw)

    def _get_pos_embed(self, height, width, device):
        num_tokens = height * width
        dim = self.transformer_dim
        half = dim // 2

        y_pos = torch.arange(
            height, device=device, dtype=torch.float32).unsqueeze(1).expand(
                height, width).reshape(-1)
        x_pos = torch.arange(
            width, device=device, dtype=torch.float32).unsqueeze(0).expand(
                height, width).reshape(-1)

        dim_t = torch.arange(half // 2, device=device, dtype=torch.float32)
        dim_t = 10000 ** (2 * dim_t / half)

        pos_embed = torch.zeros(num_tokens, dim, device=device)
        pos_embed[:, 0:half:2] = torch.sin(y_pos.unsqueeze(1) /
                                           dim_t.unsqueeze(0))
        pos_embed[:, 1:half:2] = torch.cos(y_pos.unsqueeze(1) /
                                           dim_t.unsqueeze(0))
        pos_embed[:, half::2] = torch.sin(x_pos.unsqueeze(1) /
                                          dim_t.unsqueeze(0))
        pos_embed[:, half + 1::2] = torch.cos(x_pos.unsqueeze(1) /
                                              dim_t.unsqueeze(0))
        return pos_embed.unsqueeze(0)

    def _downsample_hist_input(self, x):
        if self.hist_downsample_factor <= 1:
            return x
        height, width = x.shape[-2:]
        out_h = max(1, math.ceil(height / self.hist_downsample_factor))
        out_w = max(1, math.ceil(width / self.hist_downsample_factor))
        return F.adaptive_avg_pool2d(x, (out_h, out_w))

    def _build_soft_quantile_hist_token(self, x_8bit):
        samples = self._downsample_hist_input(
            x_8bit.clamp(0.0, 1.0)).flatten(2)
        dtype = samples.dtype
        value_grid = self.hist_value_grid.to(dtype=dtype)
        quantile_levels = self.hist_quantiles.to(dtype=dtype)

        # Smooth inverse-CDF surrogate for the paper's soft-sorted quantiles.
        values = value_grid.view(1, 1, -1, 1)
        cdf = torch.sigmoid(
            (values - samples.unsqueeze(2)) / self.hist_sigma).mean(dim=-1)

        cdf_distance = torch.abs(
            cdf.unsqueeze(2) - quantile_levels.view(1, 1, -1, 1))
        inverse_temp = max(1.0 / float(self.hist_bins), self.hist_sigma * 0.5)
        weights = torch.softmax(-cdf_distance / inverse_temp, dim=-1)
        quantiles = (weights * value_grid.view(1, 1, 1, -1)).sum(dim=-1)
        feat = quantiles.reshape(x_8bit.shape[0], -1)
        return self.hist_token_encoder(feat)

    def _build_legacy_hist_token(self, x_8bit):
        hist = self.histogram(x_8bit)
        return self.hist_encoder(hist).flatten(1)

    def _build_condition_inputs(self, x_8bit, raw_bit_depth):
        if raw_bit_depth is None:
            raise ValueError(
                'cond_injection_type is enabled but raw_bit_depth is missing. '
                'Make sure loaders write results[\'raw_bit_depth\'], '
                'PackDetInputs keeps it in meta_keys, and the detector passes '
                'batch_data_samples into the DIA backbone.')

        if self.use_hist_prefix_bit_adaln:
            return self._build_hist_prefix_bit_inputs(x_8bit, raw_bit_depth)

        bit_depth = raw_bit_depth.float().view(x_8bit.shape[0], 1)
        bit_depth = bit_depth / self.bit_depth_norm_max
        bit_token = self.bit_encoder(bit_depth)

        if self.hist_condition_mode == 'legacy_conv':
            hist_token = self._build_legacy_hist_token(x_8bit)
        else:
            hist_token = self._build_soft_quantile_hist_token(x_8bit)

        return torch.stack([bit_token, hist_token], dim=1)

    def _build_hist_prefix_bit_inputs(self, x_8bit, raw_bit_depth):
        bit_depth = raw_bit_depth.long().view(x_8bit.shape[0])
        bit_idx = bit_depth.clamp(
            min=self.bit_embed_min, max=self.bit_embed_max) - self.bit_embed_min
        bit_cond = self.bit_encoder(bit_idx)

        hist = self.histogram(x_8bit)
        hist_feat = self.hist_encoder(hist)
        hist_prefix = hist_feat.squeeze(-1).transpose(1, 2).contiguous()
        return hist_prefix, bit_cond

    def predict_grid(self, x_8bit, raw_bit_depth=None):
        batch_size = x_8bit.shape[0]

        feat = self.conv_stem(x_8bit)
        gh, gw = feat.shape[2], feat.shape[3]
        tokens = feat.flatten(2).transpose(1, 2)

        pos_embed = self._get_pos_embed(gh, gw, tokens.device)
        tokens = tokens + pos_embed

        cond_tokens = None
        bit_cond = None
        if self.use_cond_injection:
            cond_inputs = self._build_condition_inputs(x_8bit, raw_bit_depth)
            if self.use_hist_prefix_bit_adaln:
                hist_prefix, bit_cond = cond_inputs
                tokens = torch.cat([hist_prefix, tokens], dim=1)
            else:
                cond_tokens = cond_inputs

        for blk in self.transformer_blocks:
            tokens = blk(tokens, cond_tokens=cond_tokens, bit_cond=bit_cond)

        tokens = self.norm_out(tokens)

        if self.use_hist_prefix_bit_adaln:
            grid_tokens = tokens[:, self.hist_prefix_tokens:]
        else:
            grid_tokens = tokens

        if self.use_bezier:
            delta = self.curve_head(grid_tokens.mean(dim=1))
        else:
            delta = None

        feat = grid_tokens.transpose(1, 2).reshape(
            batch_size, self.transformer_dim, gh, gw)

        if self.use_grid:
            matrix_grid = self.grid_head(feat)
            matrix_grid = matrix_grid.reshape(
                batch_size, self.grid_depth, self.C_grid, gh, gw)
            matrix_grid = matrix_grid.permute(0, 2, 1, 3, 4)
            grid_out = self._reconstruct_DIA(matrix_grid)
        else:
            grid_out = None

        return grid_out, delta, (gh, gw)

    def build_control_points(self, delta):
        batch_size = delta.shape[0]
        n = self.curve_n

        if self.bezier_type == 'init_bias':
            delta = delta.view(batch_size, 3, n - 1)
            init_interior = torch.linspace(
                0, 1, steps=n + 1, device=delta.device,
                dtype=delta.dtype)[1:-1]
            init_interior = init_interior.view(1, 1, n - 1)
            bias = torch.tanh(delta) * 0.5
            p_interior = torch.clamp(init_interior + bias, 0.0, 1.0)
            p0 = torch.zeros(batch_size, 3, 1, device=delta.device,
                             dtype=delta.dtype)
            pn = torch.ones(batch_size, 3, 1, device=delta.device,
                            dtype=delta.dtype)
            p = torch.cat([p0, p_interior, pn], dim=2)
        else:
            delta = delta.view(batch_size, 3, n)
            delta = F.relu(delta) + 1e-6
            delta_hat = delta / delta.sum(dim=2, keepdim=True)
            p_rest = torch.cumsum(delta_hat, dim=2)
            p0 = torch.zeros(batch_size, 3, 1, device=delta.device,
                             dtype=delta.dtype)
            p = torch.cat([p0, p_rest], dim=2)

        return p

    def bezier_map(self, raw, control_points):
        n = self.curve_n
        raw_mapped = torch.zeros_like(raw)
        for i in range(n + 1):
            coef = math.comb(n, i)
            term = coef * ((1 - raw) ** (n - i)) * (raw ** i)
            point = control_points[:, :, i].unsqueeze(-1).unsqueeze(-1)
            raw_mapped += term * point
        return torch.clamp(raw_mapped, 0.0, 1.0)

    def slice_grid(self, grid, guide_map, full_height, full_width):
        batch_size = grid.shape[0]
        device = grid.device

        gy = torch.linspace(-1, 1, full_height, device=device)
        gx = torch.linspace(-1, 1, full_width, device=device)
        gy, gx = torch.meshgrid(gy, gx, indexing='ij')
        gy = gy.unsqueeze(0).expand(batch_size, -1, -1)
        gx = gx.unsqueeze(0).expand(batch_size, -1, -1)
        gz = guide_map.squeeze(1) * 2 - 1

        sampling_grid = torch.stack([gx, gy, gz], dim=-1)
        sampling_grid = sampling_grid.unsqueeze(1)
        coeffs = F.grid_sample(
            grid,
            sampling_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )
        return coeffs.squeeze(2)

    def apply_matrix(self, coeffs, x):
        batch_size, _, height, width = x.shape
        matrix = coeffs.reshape(batch_size, 3, 3, height, width)
        rgb = torch.einsum('bijhw,bjhw->bihw', matrix, x)
        return torch.clamp(rgb, 0.0, 1.0)

    def forward(self, x_8bit, x_12bit, raw_bit_depth=None):
        _, _, height, width = x_8bit.shape

        grid, delta, _ = self.predict_grid(
            x_8bit, raw_bit_depth=raw_bit_depth)

        if self.use_bezier:
            control_points = self.build_control_points(delta)
            x_mapped = self.bezier_map(x_12bit, control_points)

            self._fwd_count += 1
            if self.training and self._fwd_count % 200 == 0:
                with torch.no_grad():
                    delta_np = delta[0].detach().cpu().numpy()
                    p_r = control_points[0, 0].detach().cpu().numpy()
                    p_g = control_points[0, 1].detach().cpu().numpy()
                    p_b = control_points[0, 2].detach().cpu().numpy()
                    bias_np = self.curve_head.bias.detach().cpu().numpy()
                    _log_info(
                        f'[Bezier@iter{self._fwd_count}] '
                        f'raw_delta={_fmt(delta_np)} | '
                        f'ctrl_pts_R={_fmt(p_r)} | '
                        f'ctrl_pts_G={_fmt(p_g)} | '
                        f'ctrl_pts_B={_fmt(p_b)} | '
                        f'curve_head_bias={_fmt(bias_np)}'
                    )
        else:
            x_mapped = x_12bit

        if not self.use_grid:
            return x_mapped

        guide_map = x_12bit.mean(dim=1, keepdim=True)
        coeffs = self.slice_grid(grid, guide_map, height, width)
        rgb = self.apply_matrix(coeffs, x_mapped)
        return rgb
