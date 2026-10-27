#!/usr/bin/env python3
"""Generate the released rod_synspec_physbit_seed3407_full NPY images.

python tools/syn_spec/generate_rod_mixbit.py --raw-root /path/to/ROD --output-root /path/to/new_output
Add --only train/day-02000 val/day-00000 for a small hash check; keep both full RAW splits available.

Dependencies: scikit-learn==1.3.2, colour-science==0.4.1.
Verified with Python 3.10, numpy==1.26.4 and scipy==1.15.3.
Outputs are float32 in [0,1]; do not divide these generated files by 65535 again.
"""

import argparse
import json
from pathlib import Path

import numpy as np


# Published settings. Keep the full sorted train/val order and RNG call order.
SEED = 3407
BIT_DEPTHS = (8, 9, 10, 11, 12)
SOURCE_WHITE_LEVEL = 65535.0
RAW_HEIGHT, RAW_WIDTH = 1856, 2880
SPLITS = (('train', '00Train', 16089), ('val', '01Valid', 4000))
U_RANGE = (-3.0, 2.0)                 # Exposure multiplier = 2**u.
KELVIN_RANGE = (2500.0, 20000.0)      # Sample uniformly in reciprocal temperature.
TINT_RANGE = (-0.04, 0.04)
EPS_RANGE = (-0.05, 0.05)             # Off-diagonal channel mixing.
BLACK_OFFSET = 0.02
SATURATION_RANGE = (0.80, 1.00)
CAMERA_SEED_MAX = 2**31 - 1


def load_rod_raw(path):
    packed = np.fromfile(path, dtype=np.uint8)
    expected = RAW_HEIGHT * RAW_WIDTH * 3
    if packed.size != expected:
        raise ValueError(f'{path}: expected {expected} bytes, got {packed.size}')
    mosaic = (
        packed[0::3].astype(np.uint32)
        + packed[1::3].astype(np.uint32) * 256
        + packed[2::3].astype(np.uint32) * 65536
    ).reshape(RAW_HEIGHT, RAW_WIDTH)
    green = (mosaic[0::2, 1::2].astype(np.uint64)
             + mosaic[1::2, 0::2].astype(np.uint64)) // 2
    rgb = np.stack((mosaic[0::2, 0::2], green.astype(np.uint32),
                    mosaic[1::2, 1::2]), axis=-1)
    return np.clip(rgb.astype(np.float32) / SOURCE_WHITE_LEVEL, 0.0, 1.0)


def simulate_rod(image, rng, simulator, bit_depth):
    # Preserve the released ROD pipeline, including source-WB restoration.
    raw = image.astype(np.float32)
    means = raw.mean(axis=(0, 1)).astype(np.float64) + 1e-8
    wb_gain = (means[1] / means).astype(np.float32)
    balanced = raw * wb_gain.reshape(1, 1, 3)

    u = float(rng.uniform(*U_RANGE))
    alpha = float(2.0 ** u)
    mired = float(rng.uniform(1e6 / KELVIN_RANGE[1], 1e6 / KELVIN_RANGE[0]))
    kelvin = float(1e6 / max(mired, 1e-8))
    delta_uv = float(rng.uniform(*TINT_RANGE))
    illuminant = simulator._planckian_spectrum(kelvin, simulator.wavelengths)
    tint_gain = simulator._tint_to_diagonal(illuminant, delta_uv).astype(np.float32)
    camera_seed = int(rng.integers(0, CAMERA_SEED_MAX))
    camera = simulator._sample_camera(np.random.default_rng(camera_seed)).astype(np.float32)
    cross = simulator._cross_camera_matrix(camera, illuminant).astype(np.float32)
    tinted = (np.diag(tint_gain).astype(np.float32) @ cross).astype(np.float32)
    epsilon = rng.uniform(*EPS_RANGE, size=(3, 3)).astype(np.float32)
    np.fill_diagonal(epsilon, 0.0)
    beta = rng.uniform(-BLACK_OFFSET, BLACK_OFFSET, size=3).astype(np.float32)
    saturation = float(rng.uniform(*SATURATION_RANGE))
    matrix = (alpha * (tinted + epsilon)).astype(np.float32)

    out = (matrix @ balanced.reshape(-1, 3).T).T.reshape(raw.shape) + beta.reshape(1, 1, 3)
    out = out / wb_gain.reshape(1, 1, 3)
    sat = max(saturation, 1e-8)
    out = sat * np.tanh(out / sat)
    levels = float((1 << int(bit_depth)) - 1)
    out = np.clip(out, 0.0, sat)
    out = np.floor(out / sat * levels + 0.5) / levels
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    return out, {
        'raw_bit_depth': int(bit_depth), 'storage_encoding': 'normalized_float',
        'camera_seed': camera_seed, 'u': u, 'alpha': alpha,
        'kelvin': kelvin, 'mired': mired, 'delta_uv': delta_uv,
        'white_balance_gain': wb_gain.tolist(), 'matrix': matrix.tolist(),
        'beta': beta.tolist(), 'saturation': saturation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True,
                        help='Directory containing the full 00Train/ and 01Valid/ RAW splits.')
    parser.add_argument('--output-root', type=Path, required=True,
                        help='New output directory; existing directories are never overwritten.')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--only', nargs='+', metavar='SPLIT/STEM',
                        help='Save selected samples while preserving the full release RNG sequence.')
    args = parser.parse_args()

    from syn_spec import RAWSimulator

    paths = {}
    for split, folder, expected_count in SPLITS:
        paths[split] = sorted((args.raw_root / folder).glob('*.raw'))
        if len(paths[split]) != expected_count:
            raise ValueError(
                f'{folder}: expected {expected_count} RAW files, found {len(paths[split])}. '
                'Both full file lists are required for release-compatible RNG ordering; '
                'use --only to generate selected outputs.')
    available = {f'{split}/{path.stem}' for split, files in paths.items() for path in files}
    selected = set(args.only) if args.only else available
    if selected - available:
        raise ValueError(f'Unknown samples: {sorted(selected - available)}')
    rng = np.random.default_rng(args.seed)
    bit_assignments = {
        split: {path.stem: int(rng.choice(BIT_DEPTHS)) for path in files}
        for split, files in paths.items()
    }
    simulator = RAWSimulator()
    args.output_root.mkdir(parents=True, exist_ok=False)
    placeholder = np.full((1, 1, 3), 0.5, dtype=np.float32)
    metadata = {
        'seed': args.seed, 'bit_choices': list(BIT_DEPTHS),
        'source_white_level': SOURCE_WHITE_LEVEL, 'storage_encoding': 'normalized_float',
        'source_counts': {split: len(files) for split, files in paths.items()},
        'images': {},
    }
    total, written = 0, 0
    for split, files in paths.items():
        saved = []
        for path in files:
            key = f'{split}/{path.stem}'
            render = key in selected
            # Unselected images still consume identical random draws.
            image = load_rod_raw(path) if render else placeholder
            result, params = simulate_rod(image, rng, simulator, bit_assignments[split][path.stem])
            if render:
                destination = args.output_root / 'images' / split / (path.stem + '.npy')
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.save(destination, result, allow_pickle=False)
                metadata['images'][path.stem] = {'split': split, **params}
                saved.append(path.stem)
                written += 1
            total += 1
            if total % 2000 == 0:
                print(f'Processed {total}/20089; saved {written}', flush=True)
        (args.output_root / f'{split}.txt').write_text('\n'.join(saved) + '\n', encoding='utf-8')
    (args.output_root / 'metadata.json').write_text(
        json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(f'Saved {written} normalized float32 ROD images to {args.output_root}', flush=True)


if __name__ == '__main__':
    main()
