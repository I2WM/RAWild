from mmseg.registry import MODELS

from .mit import MixVisionTransformer
from .rawild_adapter.bilateral_grid_dia import BilateralGridAdapter_DIA


def _split_raw_input(x, backbone_name):
    x = x.float()
    if x.shape[1] == 6:
        return x[:, :3], x[:, 3:]
    if x.shape[1] == 3:
        return x, x
    raise ValueError(
        f'{backbone_name} expects 3 or 6 channels, got {x.shape[1]} channels.')


@MODELS.register_module()
class RAW_BilateralGrid_MixVisionTransformer_DIA(MixVisionTransformer):
    """MixVisionTransformer backbone with the RAWild DIA adapter."""

    def __init__(self,
                 in_channels=3,
                 grid_depth=8,
                 transformer_dim=128,
                 grid_num_heads=4,
                 grid_num_layers=2,
                 grid_mlp_ratio=4.0,
                 grid_drop_rate=0.0,
                 curve_n=None,
                 use_bezier=None,
                 bezier_type=None,
                 bezier_log_interval=200,
                 dia_k=0.05,
                 dia_activation='exp_tanh',
                 dia_exp_alpha=1.0,
                 dia_matrix_mode='d_ia',
                 cond_injection_type='none',
                 hist_bins=64,
                 hist_sigma=0.1,
                 hist_downsample_factor=16,
                 fixed_bit_depth=8,
                 bit_depth_norm_max=24.0,
                 **kwargs):
        if in_channels not in (3, 6):
            raise ValueError(
                'RAW_BilateralGrid_MixVisionTransformer_DIA only supports '
                f'3 or 6 input channels, got {in_channels}.')

        super().__init__(in_channels=3, **kwargs)
        self.raw_in_channels = in_channels
        self.fixed_bit_depth = float(fixed_bit_depth)
        self.bilateral_grid_adapter = BilateralGridAdapter_DIA(
            grid_depth=grid_depth,
            transformer_dim=transformer_dim,
            num_heads=grid_num_heads,
            num_layers=grid_num_layers,
            mlp_ratio=grid_mlp_ratio,
            drop_rate=grid_drop_rate,
            curve_n=curve_n,
            use_bezier=use_bezier,
            bezier_type=bezier_type,
            dia_k=dia_k,
            dia_activation=dia_activation,
            dia_exp_alpha=dia_exp_alpha,
            dia_matrix_mode=dia_matrix_mode,
            cond_injection_type=cond_injection_type,
            hist_bins=hist_bins,
            hist_sigma=hist_sigma,
            hist_downsample_factor=hist_downsample_factor,
            bit_depth_norm_max=bit_depth_norm_max,
        )

    def init_weights(self):
        super().init_weights()
        self.bilateral_grid_adapter._init_weights()

    def forward(self, x):
        x_guide, x_apply = _split_raw_input(
            x, 'RAW_BilateralGrid_MixVisionTransformer_DIA')
        raw_bit_depth = x_guide.new_full((x_guide.shape[0],),
                                         self.fixed_bit_depth)
        x = self.bilateral_grid_adapter(
            x_guide, x_apply, raw_bit_depth=raw_bit_depth)
        return super().forward(x)
