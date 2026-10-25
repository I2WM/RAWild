#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import mmcv
import numpy as np


@dataclass(frozen=True)
class ResizeOptions:
    scale_ratio: float
    save_dtype: str
    interpolation: str
    backend: str
    overwrite: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Create a ratio-resized detector cache from the '
            'full-resolution MultiRAW NPY cache.'
        ))
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--source-cache-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--scale-ratio', type=float, required=True,
                        help='Spatial scale; the original experiment uses 0.3333333333.')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--limit-per-subset', type=int, default=0)
    parser.add_argument('--subsets', nargs='*', default=None)
    parser.add_argument('--save-dtype', choices=('uint16', 'float32'), default='uint16')
    parser.add_argument('--interpolation', type=str, default='bilinear')
    parser.add_argument('--backend', choices=('cv2', 'pillow'), default='cv2')
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--progress-every', type=int, default=100)
    parser.add_argument('--job-name', type=str, default='')
    parser.add_argument('--skip-metadata-write', action='store_true')
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
            subset = row['subset']
            if subsets and subset not in subsets:
                continue
            if limit_per_subset > 0 and per_subset_counts.get(subset, 0) >= limit_per_subset:
                continue
            rows.append(row)
            per_subset_counts[subset] = per_subset_counts.get(subset, 0) + 1
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def resolve_cache_path(images_root: Path, row: dict) -> Path:
    return images_root / row['subset'] / f"{row['stem']}.npy"


def normalize_rgb(img: np.ndarray) -> np.ndarray:
    if np.issubdtype(img.dtype, np.integer):
        denom = float(np.iinfo(img.dtype).max)
        img = img.astype(np.float32) / max(denom, 1.0)
    else:
        img = img.astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def resize_rgb(img: np.ndarray,
               scale_ratio: float,
               interpolation: str,
               backend: str) -> tuple[np.ndarray, tuple[float, float]]:
    if scale_ratio <= 0:
        raise ValueError(f'scale_ratio must be > 0, got {scale_ratio}')
    old_h, old_w = img.shape[:2]
    target_w = max(1, int(round(float(old_w) * float(scale_ratio))))
    target_h = max(1, int(round(float(old_h) * float(scale_ratio))))
    resized, w_scale, h_scale = mmcv.imresize(
        img,
        (target_w, target_h),
        return_scale=True,
        interpolation=interpolation,
        backend=backend,
    )
    return resized, (float(w_scale), float(h_scale))


def save_rgb(img: np.ndarray, output_path: Path, save_dtype: str) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(img, 0.0, 1.0)
    if save_dtype == 'uint16':
        payload = np.round(img * 65535.0).astype(np.uint16)
    elif save_dtype == 'float32':
        payload = img.astype(np.float32)
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


def process_one(task: tuple[dict, str, str, ResizeOptions]) -> dict:
    row, source_cache_root_str, output_root_str, options = task
    source_cache_root = Path(source_cache_root_str)
    output_root = Path(output_root_str)
    source_path = resolve_cache_path(source_cache_root / 'images', row)
    output_path = resolve_cache_path(output_root / 'images', row)

    if output_path.exists() and not options.overwrite:
        cached = np.load(output_path, mmap_mode='r')
        return {
            'image_id': row['image_id'],
            'status': 'skipped',
            'source_path': str(source_path),
            'saved_path': str(output_path),
            'source_shape': None,
            'source_dtype': None,
            'saved_shape': list(cached.shape),
            'saved_dtype': str(cached.dtype),
            'offline_scale_factor': None,
            'resized_height': int(cached.shape[0]),
            'resized_width': int(cached.shape[1]),
        }

    source = np.load(source_path)
    normalized = normalize_rgb(source)
    resized, scale_factor = resize_rgb(
        normalized,
        scale_ratio=options.scale_ratio,
        interpolation=options.interpolation,
        backend=options.backend,
    )
    save_meta = save_rgb(resized, output_path, options.save_dtype)
    return {
        'image_id': row['image_id'],
        'status': 'written',
        'source_path': str(source_path),
        'source_shape': list(source.shape),
        'source_dtype': str(source.dtype),
        'offline_scale_factor': [float(scale_factor[0]), float(scale_factor[1])],
        'resized_height': int(resized.shape[0]),
        'resized_width': int(resized.shape[1]),
        **save_meta,
    }


def sanitize_job_name(job_name: str) -> str:
    sanitized = ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '_' for ch in job_name.strip())
    return sanitized or 'partial'


def collect_results(tasks: list[tuple[dict, str, str, ResizeOptions]],
                    num_workers: int,
                    progress_every: int,
                    job_name: str) -> list[dict]:
    total = len(tasks)
    results: list[dict] = []
    status_counter: Counter[str] = Counter()
    executor = None
    if num_workers <= 1:
        iterator = (process_one(task) for task in tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=num_workers)
        iterator = executor.map(process_one, tasks, chunksize=1)

    try:
        for index, result in enumerate(iterator, start=1):
            results.append(result)
            status_counter[result['status']] += 1
            if progress_every > 0 and (index % progress_every == 0 or index == total):
                print(json.dumps({
                    'event': 'progress',
                    'job_name': job_name or None,
                    'processed': index,
                    'total': total,
                    'written': status_counter.get('written', 0),
                    'skipped': status_counter.get('skipped', 0),
                }, ensure_ascii=False), flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return results


def main() -> None:
    args = parse_args()
    if args.scale_ratio <= 0:
        raise ValueError(f'--scale-ratio must be > 0, got {args.scale_ratio}')
    subsets = None if args.subsets is None else set(args.subsets)
    rows = load_rows(
        args.source_manifest,
        subsets=subsets,
        limit=args.limit,
        limit_per_subset=args.limit_per_subset,
    )
    if not rows:
        raise ValueError('No manifest rows selected.')

    source_cache_root = args.source_cache_root
    source_metadata_path = source_cache_root / 'metadata' / 'metadata.json'
    source_metadata = json.loads(source_metadata_path.read_text(encoding='utf-8'))
    source_images = source_metadata.get('images', source_metadata)
    if not isinstance(source_images, dict):
        raise TypeError(
            f'Source metadata at {source_metadata_path} must be a dict or contain an "images" dict.')

    output_root = args.output_root
    (output_root / 'metadata').mkdir(parents=True, exist_ok=True)
    options = ResizeOptions(
        scale_ratio=float(args.scale_ratio),
        save_dtype=args.save_dtype,
        interpolation=args.interpolation,
        backend=args.backend,
        overwrite=bool(args.overwrite),
    )
    tasks = [(row, str(source_cache_root), str(output_root), options) for row in rows]
    results = collect_results(
        tasks=tasks,
        num_workers=args.num_workers,
        progress_every=max(int(args.progress_every), 0),
        job_name=args.job_name,
    )

    result_map = {result['image_id']: result for result in results}
    metadata_payload = {
        'source_manifest': str(args.source_manifest),
        'source_cache_root': str(source_cache_root),
        'output_root': str(output_root),
        'resize': {
            'scale': None,
            'scale_ratio': float(args.scale_ratio),
            'keep_ratio': True,
            'interpolation': args.interpolation,
            'backend': args.backend,
            'save_dtype': args.save_dtype,
        },
        'images': {},
    }

    for row in rows:
        image_id = row['image_id']
        if image_id not in source_images:
            raise KeyError(f'Missing source metadata for image_id={image_id}')
        entry = dict(source_images[image_id])
        result = result_map[image_id]
        entry.update({
            'status': result['status'],
            'source_cache_path': result['source_path'],
            'saved_path': result['saved_path'],
            'saved_dtype': result['saved_dtype'],
            'saved_shape': result['saved_shape'],
            'offline_scale_factor': result['offline_scale_factor'],
            'resized_height': result['resized_height'],
            'resized_width': result['resized_width'],
            'source_shape': result['source_shape'],
            'source_dtype': result['source_dtype'],
        })
        metadata_payload['images'][image_id] = entry

    safe_job_name = sanitize_job_name(args.job_name)
    metadata_root = output_root / 'metadata'
    if args.skip_metadata_write:
        results_path = metadata_root / f'results.{safe_job_name}.json'
        results_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
        metadata_written = False
    else:
        results_path = metadata_root / 'results.json'
        results_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
        metadata_path = metadata_root / 'metadata.json'
        metadata_path.write_text(
            json.dumps(metadata_payload, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
        metadata_written = True

    subset_counter = Counter(row['subset'] for row in rows)
    print(json.dumps({
        'event': 'completed',
        'job_name': args.job_name or None,
        'selected_rows': len(rows),
        'written': sum(1 for x in results if x['status'] == 'written'),
        'skipped': sum(1 for x in results if x['status'] == 'skipped'),
        'scale_ratio': float(args.scale_ratio),
        'output_root': str(output_root),
        'results_path': str(results_path),
        'metadata_written': metadata_written,
        'subset_counts': dict(sorted(subset_counter.items())),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
