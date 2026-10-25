#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import rawpy


@dataclass(frozen=True)
class ProcessingOptions:
    save_dtype: str
    demosaic_mode: str
    wb_mode: str
    color_matrix_mode: str
    align_mode: str
    overwrite: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Prepare MultiRAW NPY arrays with the original full-resolution Bayer conversion.')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0,
                        help='Only process the first N rows. 0 means all rows.')
    parser.add_argument('--limit-per-subset', type=int, default=0,
                        help='If > 0, keep only the first N rows from each subset.')
    parser.add_argument('--subsets', nargs='*', default=None)
    parser.add_argument('--save-dtype', choices=('float32', 'uint16'), default='uint16')
    parser.add_argument('--demosaic-mode', choices=('opencv_fullres',), default='opencv_fullres')
    parser.add_argument('--wb-mode', choices=('none', 'metadata'), default='none')
    parser.add_argument('--color-matrix-mode', choices=('none', 'metadata'), default='none')
    parser.add_argument(
        '--align-mode',
        choices=('none', 'center_crop_to_annotation', 'from_manifest'),
        default='none',
        help='Use from_manifest for the original per-sensor annotation alignment.')
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def load_rows(path: Path,
              subsets: set[str] | None,
              limit: int,
              limit_per_subset: int) -> list[dict]:
    rows: list[dict] = []
    per_subset_counts: dict[str, int] = {}
    with path.open('r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if subsets and row['subset'] not in subsets:
                continue
            subset = row['subset']
            if limit_per_subset > 0 and per_subset_counts.get(subset, 0) >= limit_per_subset:
                continue
            rows.append(row)
            per_subset_counts[subset] = per_subset_counts.get(subset, 0) + 1
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def load_raw_mosaic(row: dict) -> np.ndarray:
    raw_path = Path(row['raw_path'])
    suffix = row['raw_format'].lower()
    if suffix == '.raw.tif':
        return iio.imread(raw_path).astype(np.float32)
    with rawpy.imread(str(raw_path)) as raw:
        return raw.raw_image_visible.copy().astype(np.float32)


def normalize_raw(row: dict, mosaic: np.ndarray) -> np.ndarray:
    black = float(row['black_level'] or 0.0)
    white = float(row['white_level'])
    denom = max(white - black, 1.0)
    return np.clip((mosaic - black) / denom, 0.0, 1.0).astype(np.float32)


def demosaic_fullres(mosaic: np.ndarray, pattern: str) -> np.ndarray:
    pattern = pattern.lower()
    # Preserve the original OpenCV codes: standard Bayer names yield BGR channel order.
    code_map = {
        'rggb': cv2.COLOR_BayerRG2RGB,
        'grbg': cv2.COLOR_BayerGR2RGB,
        'gbrg': cv2.COLOR_BayerGB2RGB,
        'bggr': cv2.COLOR_BayerBG2RGB,
    }
    code = code_map.get(pattern)
    if code is None:
        raise ValueError(f'Unsupported Bayer pattern: {pattern}')
    bayer_u16 = np.round(np.clip(mosaic, 0.0, 1.0) * 65535.0).astype(np.uint16)
    rgb_u16 = cv2.cvtColor(bayer_u16, code)
    return rgb_u16.astype(np.float32) / 65535.0


def apply_white_balance(rgb: np.ndarray, row: dict, wb_mode: str) -> np.ndarray:
    if wb_mode == 'none':
        return rgb
    wb = np.asarray(row['white_balance'], dtype=np.float32)
    if wb.shape != (3,):
        raise ValueError(f'Invalid white balance for {row["image_id"]}: {wb}')
    return rgb * wb.reshape(1, 1, 3)


def apply_color_matrix(rgb: np.ndarray, row: dict, color_matrix_mode: str) -> np.ndarray:
    if color_matrix_mode == 'none':
        return rgb
    matrix = np.asarray(row['color_matrix'], dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError(f'Invalid color matrix for {row["image_id"]}: shape={matrix.shape}')
    return np.tensordot(rgb, matrix.T, axes=1)


def align_rgb(rgb: np.ndarray, row: dict, align_mode: str) -> np.ndarray:
    original_mode = align_mode
    if align_mode == 'none':
        return rgb, 'none'
    if align_mode == 'from_manifest':
        align_mode = row.get('default_align_mode', 'none')
        if align_mode == 'none':
            return rgb, 'none'
    target_h = int(row['height'])
    target_w = int(row['width'])
    src_h, src_w = rgb.shape[:2]
    if target_h > src_h or target_w > src_w:
        raise ValueError(
            f'Cannot align {row["image_id"]}: target {(target_h, target_w)} '
            f'is larger than source {(src_h, src_w)}')
    if target_h == src_h and target_w == src_w:
        return rgb, align_mode if original_mode != 'none' else 'none'
    if align_mode == 'center_crop_to_annotation':
        top = (src_h - target_h) // 2
        left = (src_w - target_w) // 2
        return rgb[top:top + target_h, left:left + target_w], align_mode
    raise ValueError(f'Unsupported align mode: {align_mode}')


def save_rgb(rgb: np.ndarray, output_path: Path, save_dtype: str) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.clip(rgb, 0.0, 1.0)
    if save_dtype == 'float32':
        payload = rgb.astype(np.float32)
    elif save_dtype == 'uint16':
        payload = np.round(rgb * 65535.0).astype(np.uint16)
    else:
        raise ValueError(f'Unsupported save dtype: {save_dtype}')
    np.save(output_path, payload, allow_pickle=False)
    return {
        'saved_path': str(output_path),
        'saved_dtype': str(payload.dtype),
        'saved_shape': list(payload.shape),
        'saved_min': float(payload.min()),
        'saved_max': float(payload.max()),
    }


def process_one(task: tuple[dict, str, ProcessingOptions]) -> dict:
    row, output_root_str, options = task
    output_root = Path(output_root_str)
    output_path = output_root / row['subset'] / f'{row["stem"]}.npy'
    if output_path.exists() and not options.overwrite:
        return {
            'image_id': row['image_id'],
            'status': 'skipped',
            'saved_path': str(output_path),
        }

    mosaic = load_raw_mosaic(row)
    normalized = normalize_raw(row, mosaic)
    if options.demosaic_mode == 'opencv_fullres':
        rgb = demosaic_fullres(normalized, row['bayer_pattern'])
    else:
        raise ValueError(f'Unsupported demosaic mode: {options.demosaic_mode}')
    rgb = apply_white_balance(rgb, row, options.wb_mode)
    rgb = apply_color_matrix(rgb, row, options.color_matrix_mode)
    rgb, resolved_align_mode = align_rgb(rgb, row, options.align_mode)
    save_meta = save_rgb(rgb, output_path, options.save_dtype)

    return {
        'image_id': row['image_id'],
        'status': 'written',
        'decoded_shape': list(mosaic.shape),
        'aligned_shape': list(rgb.shape),
        'processing': {
            'save_dtype': options.save_dtype,
            'demosaic_mode': options.demosaic_mode,
            'wb_mode': options.wb_mode,
            'color_matrix_mode': options.color_matrix_mode,
            'align_mode': options.align_mode,
            'resolved_align_mode': resolved_align_mode,
        },
        **save_meta,
    }


def main() -> None:
    args = parse_args()
    subsets = None if args.subsets is None else set(args.subsets)
    rows = load_rows(
        args.manifest,
        subsets=subsets,
        limit=args.limit,
        limit_per_subset=args.limit_per_subset)
    if not rows:
        raise ValueError('No manifest rows selected.')

    output_root = args.output_root
    metadata_dir = output_root / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    options = ProcessingOptions(
        save_dtype=args.save_dtype,
        demosaic_mode=args.demosaic_mode,
        wb_mode=args.wb_mode,
        color_matrix_mode=args.color_matrix_mode,
        align_mode=args.align_mode,
        overwrite=bool(args.overwrite))

    tasks = [(row, str(output_root / 'images'), options) for row in rows]
    if args.num_workers <= 1:
        results = [process_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            results = list(executor.map(process_one, tasks, chunksize=1))

    metadata_payload = {
        'manifest': str(args.manifest),
        'output_root': str(output_root),
        'processing': {
            'save_dtype': args.save_dtype,
            'demosaic_mode': args.demosaic_mode,
            'wb_mode': args.wb_mode,
            'color_matrix_mode': args.color_matrix_mode,
            'align_mode': args.align_mode,
        },
        'images': {
            row['image_id']: {
                **row,
                **{k: v for k, v in result.items() if k not in ('image_id', 'status')},
            }
            for row, result in zip(rows, results)
        },
    }
    (metadata_dir / 'metadata.json').write_text(
        json.dumps(metadata_payload, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    (metadata_dir / 'results.json').write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    print(json.dumps({
        'processed': len(results),
        'written': sum(1 for x in results if x['status'] == 'written'),
        'skipped': sum(1 for x in results if x['status'] == 'skipped'),
        'output_root': str(output_root),
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
