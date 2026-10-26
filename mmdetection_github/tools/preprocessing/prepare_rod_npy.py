#!/usr/bin/env python3
"""Convert the public ICCV 2023 ROD challenge release (16089 train / 4000 val).

Dataset index:
https://github.com/amumu1843/Robust_Raw_Object_Detection/blob/main/ROD_Dataset/ROD_Dataset.md

Outputs contain unnormalized RGB uint32 sensor counts. RAWild's ROD loader
intentionally divides by 65535 once; values may exceed 1 after scaling.
Pre-normalized [0, 1] NPY files are not interchangeable with these outputs.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


RAW_HEIGHT = 1856
RAW_WIDTH = 2880
RAW_PIXEL_COUNT = RAW_HEIGHT * RAW_WIDTH
PACKED_VALUE_BYTES = 3
RAW_FILE_BYTES = RAW_PIXEL_COUNT * PACKED_VALUE_BYTES
OUT_HEIGHT = RAW_HEIGHT // 2
OUT_WIDTH = RAW_WIDTH // 2
SMOKE_SEED = 3407
SMOKE_QUOTAS = {
    'train': {'day': 25, 'night': 75},
    'val': {'day': 20, 'night': 20},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Pack 24-bit ROD RGGB mosaics into half-resolution RGB uint32 NPY files.')
    parser.add_argument(
        '--train-raw-dir',
        type=Path,
        required=True,
        help='Directory containing training .raw files.')
    parser.add_argument(
        '--val-raw-dir',
        type=Path,
        required=True,
        help='Directory containing validation .raw files.')
    parser.add_argument(
        '--output-root',
        type=Path,
        required=True,
        help='Root directory for generated smoke/full datasets.')
    parser.add_argument(
        '--variant',
        choices=('smoke', 'full', 'both'),
        default='full',
        help='Which dataset variant(s) to generate.')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help='Worker count for raw -> npy conversion.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing .npy files.')
    return parser.parse_args()


def gather_stems(raw_dir: Path) -> list[str]:
    stems = sorted(path.stem for path in raw_dir.glob('*.raw'))
    if not stems:
        raise FileNotFoundError(f'No .raw files found under {raw_dir}')
    return stems


def select_smoke_subset(stems: list[str], split_name: str) -> list[str]:
    quotas = SMOKE_QUOTAS[split_name]
    grouped = {
        'day': [stem for stem in stems if stem.startswith('day-')],
        'night': [stem for stem in stems if stem.startswith('night-')],
    }
    selected: list[str] = []
    for domain, quota in quotas.items():
        candidates = grouped[domain]
        if len(candidates) < quota:
            raise ValueError(
                f'{split_name} {domain} only has {len(candidates)} files, but quota is {quota}.')
        rng = random.Random(f'{SMOKE_SEED}:{split_name}:{domain}')
        selected.extend(rng.sample(candidates, quota))
    return sorted(selected)


def unpack_raw24(path: Path) -> np.ndarray:
    packed = np.fromfile(path, dtype=np.uint8)
    if packed.size != RAW_FILE_BYTES:
        raise ValueError(
            f'{path} has {packed.size} bytes, expected {RAW_FILE_BYTES} bytes '
            f'for {RAW_HEIGHT}x{RAW_WIDTH} packed 24-bit raw.')
    raw = (
        packed[0::3].astype(np.uint32)
        + packed[1::3].astype(np.uint32) * 256
        + packed[2::3].astype(np.uint32) * 65536
    )
    return raw.reshape((RAW_HEIGHT, RAW_WIDTH))


def demosaic_rgb(raw: np.ndarray) -> np.ndarray:
    red = raw[0::2, 0::2]
    green = (
        raw[0::2, 1::2].astype(np.uint64)
        + raw[1::2, 0::2].astype(np.uint64)
    ) // 2
    blue = raw[1::2, 1::2]
    rgb = np.stack((red, green.astype(np.uint32), blue), axis=-1)
    return rgb.astype(np.uint32, copy=False)


def convert_one(task: tuple[Path, Path, bool]) -> tuple[str, bool]:
    raw_path, npy_path, overwrite = task
    if npy_path.exists() and not overwrite:
        return raw_path.stem, False
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    raw = unpack_raw24(raw_path)
    rgb = demosaic_rgb(raw)
    np.save(npy_path, rgb, allow_pickle=False)
    return raw_path.stem, True


def export_subset(
    raw_dir: Path,
    output_dir: Path,
    stems: list[str],
    num_workers: int,
    overwrite: bool,
) -> dict[str, int]:
    tasks = [
        (raw_dir / f'{stem}.raw', output_dir / f'{stem}.npy', overwrite)
        for stem in stems
    ]
    converted = 0
    skipped = 0
    if num_workers <= 1:
        for task in tasks:
            _, wrote = convert_one(task)
            converted += int(wrote)
            skipped += int(not wrote)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for _, wrote in executor.map(convert_one, tasks, chunksize=8):
                converted += int(wrote)
                skipped += int(not wrote)
    return {'converted': converted, 'skipped': skipped, 'total': len(tasks)}


def write_metadata(dataset_root: Path, split_to_stems: dict[str, list[str]]) -> None:
    metadata_dir = dataset_root / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'seed': SMOKE_SEED,
        'raw_height': RAW_HEIGHT,
        'raw_width': RAW_WIDTH,
        'output_height': OUT_HEIGHT,
        'output_width': OUT_WIDTH,
        'splits': {
            split: {
                'count': len(stems),
                'stems': stems,
            }
            for split, stems in split_to_stems.items()
        },
    }
    (metadata_dir / 'splits.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')


def build_variant(
    variant_name: str,
    output_root: Path,
    train_raw_dir: Path,
    val_raw_dir: Path,
    train_stems: list[str],
    val_stems: list[str],
    num_workers: int,
    overwrite: bool,
) -> dict[str, dict[str, int]]:
    dataset_root = output_root / variant_name
    split_to_stems = {'train': train_stems, 'val': val_stems}
    write_metadata(dataset_root, split_to_stems)

    stats = {}
    stats['train'] = export_subset(
        train_raw_dir, dataset_root / 'images' / 'train', train_stems, num_workers, overwrite)
    stats['val'] = export_subset(
        val_raw_dir, dataset_root / 'images' / 'val', val_stems, num_workers, overwrite)
    return stats


def main() -> None:
    args = parse_args()
    train_stems_all = gather_stems(args.train_raw_dir)
    val_stems_all = gather_stems(args.val_raw_dir)

    variants: list[tuple[str, list[str], list[str]]] = []
    if args.variant in ('smoke', 'both'):
        variants.append((
            'smoke',
            select_smoke_subset(train_stems_all, 'train'),
            select_smoke_subset(val_stems_all, 'val'),
        ))
    if args.variant in ('full', 'both'):
        variants.append(('full', train_stems_all, val_stems_all))

    run_summary: dict[str, dict[str, dict[str, int]]] = {}
    for variant_name, train_stems, val_stems in variants:
        run_summary[variant_name] = build_variant(
            variant_name=variant_name,
            output_root=args.output_root,
            train_raw_dir=args.train_raw_dir,
            val_raw_dir=args.val_raw_dir,
            train_stems=train_stems,
            val_stems=val_stems,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
        )

    print(json.dumps(run_summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
