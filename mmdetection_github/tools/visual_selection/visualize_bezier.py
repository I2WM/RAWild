#!/usr/bin/env python3
"""Export RAWild adapter Bezier/grid visualizations for one dataset image."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.utils import import_modules_from_strings
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmdet.registry import DATASETS, MODELS  # noqa: E402
from mmdet.utils import register_all_modules  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run a RAWild checkpoint on one image from a config dataset and '
            'export Bezier+grid output plus the pure Bezier response curve.'
        )
    )
    parser.add_argument('--config', required=True, help='MMDetection config path.')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path.')
    parser.add_argument(
        '--image-key',
        help='Image id/stem to load from the selected dataset, e.g. 2014_000001.',
    )
    parser.add_argument(
        '--image-path',
        help='Optional image path. Its filename stem is used as image-key.',
    )
    parser.add_argument(
        '--image-index',
        type=int,
        help='Dataset index to visualize. Used only when image-key/path is omitted.',
    )
    parser.add_argument('--out-dir', required=True, help='Output directory.')
    parser.add_argument(
        '--split',
        choices=('test', 'val', 'train'),
        default='test',
        help='Which dataloader in the config to use.',
    )
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--curve-resolution', type=int, default=1024)
    parser.add_argument('--figure-size', type=float, default=8.0)
    parser.add_argument('--dpi', type=int, default=220)
    parser.add_argument(
        '--curve-only',
        action='store_true',
        help='Only save bezier_curve.png and metadata.json.',
    )
    return parser.parse_args()


def setup_registry(cfg: Config) -> None:
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg.custom_imports)
    register_all_modules()
    init_default_scope(cfg.get('default_scope', 'mmdet'))


def dataloader_cfg(cfg: Config, split: str):
    key = f'{split}_dataloader'
    if key not in cfg or cfg[key] is None:
        raise KeyError(f'Config does not define {key}.')
    loader_cfg = cfg[key].copy()
    loader_cfg.batch_size = 1
    loader_cfg.num_workers = 0
    loader_cfg.persistent_workers = False
    return loader_cfg


def build_dataset(cfg: Config, split: str):
    return DATASETS.build(dataloader_cfg(cfg, split).dataset)


def build_model(cfg: Config, checkpoint: str, device: torch.device):
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint, map_location='cpu')
    model.cfg = cfg
    model.to(device)
    model.eval()
    return model


def image_key_from_args(args: argparse.Namespace) -> Optional[str]:
    if args.image_key:
        return args.image_key
    if args.image_path:
        return Path(args.image_path).stem
    return None


def data_info_key(info: Dict) -> str:
    path = info.get('img_path') or info.get('file_name')
    if path is None:
        raise KeyError(f'Dataset data_info has no img_path/file_name: {info.keys()}')
    return Path(path).stem


def find_dataset_index(dataset, image_key: Optional[str], image_index: Optional[int]) -> int:
    if image_key is None:
        if image_index is None:
            raise ValueError('Provide --image-key, --image-path, or --image-index.')
        if image_index < 0 or image_index >= len(dataset):
            raise IndexError(f'--image-index={image_index} outside dataset length {len(dataset)}.')
        return image_index

    for idx in range(len(dataset)):
        if data_info_key(dataset.get_data_info(idx)) == image_key:
            return idx
    raise KeyError(f'Image key {image_key!r} was not found in the selected dataset.')


def get_processed_inputs(model, data):
    processed = model.data_preprocessor(data, training=False)
    if not isinstance(processed, dict):
        raise TypeError(f'Unexpected data_preprocessor output: {type(processed)}')
    return processed['inputs'].float(), processed.get('data_samples', data['data_samples'])


def sample_metainfo(sample) -> Dict:
    if hasattr(sample, 'metainfo'):
        return sample.metainfo
    if hasattr(sample, 'metainfo_items'):
        return dict(sample.metainfo_items())
    return sample


def collect_raw_bit_depth(samples: Iterable, device: torch.device, dtype: torch.dtype):
    values = []
    for idx, sample in enumerate(samples):
        meta = sample_metainfo(sample)
        value = meta.get('raw_bit_depth')
        if value is None:
            raise KeyError(f'Missing raw_bit_depth for sample index {idx}.')
        values.append(float(value))
    return torch.tensor(values, device=device, dtype=dtype).view(-1, 1)


def adapter_predict_grid(adapter, x_guide, raw_bit_depth):
    if raw_bit_depth is not None:
        try:
            return adapter.predict_grid(x_guide, raw_bit_depth=raw_bit_depth)
        except TypeError:
            pass
    return adapter.predict_grid(x_guide)


def adapter_forward(adapter, x_guide, x_apply, raw_bit_depth):
    if raw_bit_depth is not None:
        try:
            return adapter(x_guide, x_apply, raw_bit_depth=raw_bit_depth)
        except TypeError:
            pass
    return adapter(x_guide, x_apply)


def crop_to_img_shape(tensor: torch.Tensor, img_shape: Tuple[int, int]) -> torch.Tensor:
    height, width = int(img_shape[0]), int(img_shape[1])
    return tensor[:, :height, :width]


def chw_to_rgb_uint8(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().float().clamp(0.0, 1.0).numpy()
    array = np.transpose(array[:3], (1, 2, 0))
    return np.rint(array * 255.0).astype(np.uint8)


def save_png(tensor: torch.Tensor, path: Path) -> None:
    Image.fromarray(chw_to_rgb_uint8(tensor), mode='RGB').save(path)


def plot_bezier_curve(x_values: np.ndarray, curve_rgb: Dict[str, np.ndarray], out_path: Path,
                      figure_size: float, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(figure_size, figure_size), dpi=dpi)
    ax.plot(x_values, x_values, linestyle='--', dashes=(8, 6),
            color='#b0b0b0', linewidth=2.0)
    ax.plot(x_values, curve_rgb['r'], color='#ff0000', linewidth=3.0)
    ax.plot(x_values, curve_rgb['g'], color='#00aa00', linewidth=3.0)
    ax.plot(x_values, curve_rgb['b'], color='#0066ff', linewidth=3.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    fig.savefig(out_path, dpi=dpi, facecolor='white')
    plt.close(fig)


def tensor_stats(tensor: torch.Tensor) -> Dict:
    tensor = tensor.detach().cpu().float().clamp(0.0, 1.0)
    return {
        'shape': [int(v) for v in tensor.shape],
        'min': float(tensor.min().item()),
        'max': float(tensor.max().item()),
        'mean': float(tensor.mean().item()),
        'std': float(tensor.std(unbiased=False).item()),
    }


def tensor_list(tensor: Optional[torch.Tensor]):
    if tensor is None:
        return None
    return tensor.detach().cpu().float().tolist()


def export_visuals(args: argparse.Namespace) -> Dict:
    cfg = Config.fromfile(args.config)
    setup_registry(cfg)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith('cuda') else 'cpu')
    dataset = build_dataset(cfg, args.split)
    idx = find_dataset_index(dataset, image_key_from_args(args), args.image_index)
    data_info = dataset.get_data_info(idx)

    model = build_model(cfg, args.checkpoint, device)
    adapter = getattr(model.backbone, 'bilateral_grid_adapter', None)
    if adapter is None:
        raise AttributeError('model.backbone has no bilateral_grid_adapter.')

    data = pseudo_collate([dataset[idx]])
    inputs, data_samples = get_processed_inputs(model, data)
    meta = sample_metainfo(data_samples[0])
    img_shape = tuple(meta.get('img_shape', inputs.shape[-2:])[:2])

    raw_bit_depth = None
    if getattr(model.backbone, 'needs_data_samples', False) or getattr(model.backbone, 'needs_img_metas', False):
        raw_bit_depth = collect_raw_bit_depth(data_samples, device=inputs.device, dtype=inputs.dtype)

    x_guide = inputs[:, :3]
    x_apply = inputs[:, 3:]
    if x_apply.shape[1] != 3:
        raise ValueError(f'Expected 6-channel input split into 3+3, got {tuple(inputs.shape)}.')

    with torch.no_grad():
        grid_out, delta, grid_shape = adapter_predict_grid(adapter, x_guide, raw_bit_depth)

        control_points = None
        if delta is not None and bool(getattr(adapter, 'use_bezier', False)):
            control_points = adapter.build_control_points(delta)
            bezier_only = adapter.bezier_map(x_apply, control_points).clamp(0.0, 1.0)
        else:
            bezier_only = x_apply.clamp(0.0, 1.0)

        bezier_grid = adapter_forward(adapter, x_guide, x_apply, raw_bit_depth).clamp(0.0, 1.0)

        dense_x = torch.linspace(0.0, 1.0, steps=args.curve_resolution,
                                 device=inputs.device, dtype=inputs.dtype)
        if control_points is None:
            dense_output = dense_x.view(1, 1, 1, -1).expand(1, 3, 1, -1)
        else:
            dense_input = dense_x.view(1, 1, 1, -1).expand(1, 3, 1, -1)
            dense_output = adapter.bezier_map(dense_input, control_points).clamp(0.0, 1.0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dense_x_np = dense_x.detach().cpu().float().numpy()
    dense_output_np = dense_output[0, :, 0, :].detach().cpu().float().numpy()
    curve_rgb = {
        'r': dense_output_np[0],
        'g': dense_output_np[1],
        'b': dense_output_np[2],
    }

    curve_path = out_dir / 'bezier_curve.png'
    plot_bezier_curve(dense_x_np, curve_rgb, curve_path, args.figure_size, args.dpi)

    linear = crop_to_img_shape(x_apply[0].clamp(0.0, 1.0), img_shape)
    bezier_only = crop_to_img_shape(bezier_only[0], img_shape)
    bezier_grid = crop_to_img_shape(bezier_grid[0], img_shape)

    output_files = {'bezier_curve': str(curve_path)}
    if not args.curve_only:
        linear_path = out_dir / 'linear_input.png'
        bezier_only_path = out_dir / 'bezier_only.png'
        bezier_grid_path = out_dir / 'bezier_grid.png'
        save_png(linear, linear_path)
        save_png(bezier_only, bezier_only_path)
        save_png(bezier_grid, bezier_grid_path)
        output_files.update({
            'linear_input': str(linear_path),
            'bezier_only': str(bezier_only_path),
            'bezier_grid': str(bezier_grid_path),
        })

    metadata = {
        'config': str(Path(args.config).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'split': args.split,
        'dataset_index': int(idx),
        'image_key': data_info_key(data_info),
        'img_path': data_info.get('img_path') or data_info.get('file_name'),
        'img_shape': [int(img_shape[0]), int(img_shape[1])],
        'padded_input_shape': [int(inputs.shape[2]), int(inputs.shape[3])],
        'raw_bit_depth': None if raw_bit_depth is None else float(raw_bit_depth[0, 0].item()),
        'adapter': {
            'class': adapter.__class__.__name__,
            'use_bezier': bool(getattr(adapter, 'use_bezier', False)),
            'use_grid': bool(getattr(adapter, 'use_grid', False)),
            'bezier_type': str(getattr(adapter, 'bezier_type', 'none')),
            'curve_n': int(getattr(adapter, 'curve_n', 0)),
            'grid_shape': None if grid_shape is None else [int(v) for v in grid_shape],
            'grid_out_shape': None if grid_out is None else [int(v) for v in grid_out.shape],
        },
        'delta': None if delta is None else {
            'shape': [int(v) for v in delta.shape],
            'values': tensor_list(delta[0]),
        },
        'control_points': None if control_points is None else tensor_list(control_points[0]),
        'stats': {
            'linear_input': tensor_stats(linear),
            'bezier_only': tensor_stats(bezier_only),
            'bezier_grid': tensor_stats(bezier_grid),
        },
        'output_files': output_files,
    }
    metadata_path = out_dir / 'metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def main() -> None:
    metadata = export_visuals(parse_args())
    print(json.dumps(metadata['output_files'], indent=2))


if __name__ == '__main__':
    main()
