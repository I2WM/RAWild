#!/usr/bin/env python3
import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
VENDOR_DIR = THIS_DIR / '_vendor'
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))
sys.path.insert(0, str(THIS_DIR))

from syn_spec import RAWSimulator


DEFAULT_SOURCE_ROOT = Path('data/cgj_simulation/demosaic_normal_12bit')
DEFAULT_TRAIN_LIST = Path('data/PASCAL_RAW/trainval/train.txt')
DEFAULT_VAL_LIST = Path('data/PASCAL_RAW/trainval/val.txt')
DEFAULT_OUTPUT_ROOT = Path('work_dirs/syn_spec/pascal_normal_mixbit')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate syn_spec Pascal-normal mixed-bit npy files.'
    )
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument('--train-list', type=Path, default=DEFAULT_TRAIN_LIST)
    parser.add_argument('--val-list', type=Path, default=DEFAULT_VAL_LIST)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--train-count', type=int, default=None)
    parser.add_argument('--val-count', type=int, default=None)
    parser.add_argument('--seed', type=int, default=20260415)
    parser.add_argument('--raw-max-value', type=float, default=4095.0)
    parser.add_argument('--bit-depths', type=int, nargs='+', default=[8, 9, 10, 11, 12])
    parser.add_argument('--u-range', type=float, nargs=2, default=[-1.5, 1.5])
    parser.add_argument('--m-range', type=float, nargs=2, default=[100.0, 250.0])
    parser.add_argument('--tint-range', type=float, nargs=2, default=[-0.02, 0.02])
    parser.add_argument('--eps-range', type=float, nargs=2, default=[-0.03, 0.03])
    parser.add_argument('--delta-bl', type=float, default=0.02)
    parser.add_argument('--s-sat-range', type=float, nargs=2, default=[0.85, 1.0])
    parser.add_argument('--cct-amplify', type=float, default=1.0)
    parser.add_argument('--auto-wb', dest='auto_wb', action='store_true')
    parser.add_argument('--no-auto-wb', dest='auto_wb', action='store_false')
    parser.set_defaults(auto_wb=True)
    return parser.parse_args()


def read_id_list(path):
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not ids:
        raise ValueError(f'No image ids found in {path}')
    return ids


def limit_ids(ids, limit):
    if limit is None:
        return ids
    if limit < 0:
        raise ValueError('Count limit must be non-negative.')
    return ids[:limit]


def build_balanced_bits(count, bit_depths, rng):
    if count == 0:
        return []
    repeats = math.ceil(count / len(bit_depths))
    values = list(bit_depths) * repeats
    rng.shuffle(values)
    return values[:count]


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(val) for val in value]
    return value


def save_text_list(path, values):
    path.write_text('\n'.join(values) + '\n')


def verify_inputs(args):
    for path in (args.source_root, args.train_list, args.val_list):
        if not path.exists():
            raise FileNotFoundError(f'Missing required path: {path}')
    if args.output_root.exists():
        raise FileExistsError(
            f'Output root already exists: {args.output_root}. Use a new path.'
        )
    if not args.bit_depths:
        raise ValueError('At least one bit depth is required.')


def main():
    args = parse_args()
    verify_inputs(args)

    train_ids = limit_ids(read_id_list(args.train_list), args.train_count)
    val_ids = limit_ids(read_id_list(args.val_list), args.val_count)

    args.output_root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    simulator = RAWSimulator()

    metadata = {
        'generated_at': datetime.now(timezone.utc).astimezone().isoformat(),
        'source_root': str(args.source_root),
        'output_root': str(args.output_root),
        'train_list': str(args.train_list),
        'val_list': str(args.val_list),
        'seed': args.seed,
        'raw_max_value': args.raw_max_value,
        'bit_depths_requested': [int(bit) for bit in args.bit_depths],
        'ranges': {
            'u_range': list(args.u_range),
            'm_range': list(args.m_range),
            'tint_range': list(args.tint_range),
            'eps_range': list(args.eps_range),
            'delta_bl': args.delta_bl,
            's_sat_range': list(args.s_sat_range),
            'cct_amplify': args.cct_amplify,
            'auto_wb': args.auto_wb,
        },
        'images': {},
    }
    split_histograms = {}

    for split_name, image_ids in (('train', train_ids), ('val', val_ids)):
        assigned_bits = build_balanced_bits(len(image_ids), args.bit_depths, rng)
        split_hist = Counter()
        for image_id, forced_bit in zip(image_ids, assigned_bits):
            src_path = args.source_root / f'{image_id}.npy'
            if not src_path.exists():
                raise FileNotFoundError(f'Missing source npy: {src_path}')

            raw = np.load(src_path).astype(np.float32) / args.raw_max_value
            if raw.ndim != 3 or raw.shape[2] != 3:
                raise ValueError(f'Unexpected shape for {src_path}: {raw.shape}')

            augmented, params = simulator.augment(
                raw,
                rng=rng,
                u_range=tuple(args.u_range),
                m_range=tuple(args.m_range),
                tint_range=tuple(args.tint_range),
                eps_range=tuple(args.eps_range),
                delta_bl=args.delta_bl,
                s_sat_range=tuple(args.s_sat_range),
                bit_depths=(int(forced_bit),),
                auto_wb=args.auto_wb,
                cct_amplify=args.cct_amplify,
            )
            dst_path = args.output_root / f'{image_id}.npy'
            np.save(dst_path, augmented.astype(np.float32))

            bit_depth = int(params['bit_depth'])
            split_hist[bit_depth] += 1
            metadata['images'][image_id] = {
                'split': split_name,
                'source_path': str(src_path),
                'output_path': str(dst_path),
                'raw_bit_depth': bit_depth,
                'raw_shape': list(raw.shape),
                'output_dtype': str(augmented.dtype),
                'output_min': float(augmented.min()),
                'output_max': float(augmented.max()),
                'output_mean': float(augmented.mean()),
                'forced_bit_depth': int(forced_bit),
                'params': to_jsonable(params),
            }
        split_histograms[split_name] = {str(bit): split_hist[bit] for bit in sorted(split_hist)}

    save_text_list(args.output_root / 'train.txt', train_ids)
    save_text_list(args.output_root / 'val.txt', val_ids)

    summary = {
        'counts': {
            'train': len(train_ids),
            'val': len(val_ids),
            'total': len(train_ids) + len(val_ids),
        },
        'bit_depth_histogram': {
            'train': split_histograms.get('train', {}),
            'val': split_histograms.get('val', {}),
            'all': {
                str(bit): int(split_histograms.get('train', {}).get(str(bit), 0))
                + int(split_histograms.get('val', {}).get(str(bit), 0))
                for bit in sorted(set(int(bit) for bit in args.bit_depths))
            },
        },
        'sample_ids': {
            'train': train_ids[:5],
            'val': val_ids[:5],
        },
    }

    (args.output_root / 'metadata.json').write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    (args.output_root / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f'Generated dataset at: {args.output_root}')


if __name__ == '__main__':
    main()
