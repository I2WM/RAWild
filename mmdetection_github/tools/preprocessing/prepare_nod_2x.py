#!/usr/bin/env python3
"""Build NOD RGGB arrays at twice the RAM annotation resolution."""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


SCALE = 2
DATASETS = {
    'nod-nikon': dict(
        source='sources/NOD_RAW_Nikon',
        suffix='.NEF',
        annotations=(
            'raw_new_Nikon750_train.json',
            'raw_new_Nikon750_test.json'),
        destination=lambda split: f'NOD_2x/raw_new_Nikon750_{split}',
    ),
    'nod-sony': dict(
        source='sources/NOD_RAW_Sony',
        suffix='.ARW',
        annotations=(
            'raw_new_Sony_RX100m7_train.json',
            'raw_new_Sony_RX100m7_test.json'),
        destination=lambda split: f'NOD_2x/raw_new_Sony_RX100m7_{split}',
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage-root', type=Path, required=True,
                        help='Root containing sources/ and annotations/NOD/.')
    parser.add_argument(
        '--dataset', action='append', required=True, choices=tuple(DATASETS))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def scale_annotation(source, target):
    payload = json.loads(source.read_text())
    for image in payload['images']:
        image['height'] = int(image['height']) * SCALE
        image['width'] = int(image['width']) * SCALE
    for annotation in payload['annotations']:
        annotation['bbox'] = [float(value) * SCALE
                              for value in annotation['bbox']]
        if 'area' in annotation:
            annotation['area'] = float(annotation['area']) * SCALE * SCALE
        segmentation = annotation.get('segmentation')
        if segmentation:
            if not isinstance(segmentation, list):
                raise ValueError(
                    f'Unsupported RLE segmentation in {source}: '
                    f"annotation {annotation.get('id')}")
            annotation['segmentation'] = [
                [float(value) * SCALE for value in polygon]
                for polygon in segmentation
            ]
        keypoints = annotation.get('keypoints')
        if keypoints:
            annotation['keypoints'] = [
                float(value) * SCALE if index % 3 != 2 else value
                for index, value in enumerate(keypoints)
            ]
    atomic_json(target, payload)
    return payload


def annotation_split(path):
    if 'train' in path.stem.lower():
        return 'train'
    if 'test' in path.stem.lower():
        return 'test'
    raise ValueError(f'Cannot infer split from {path}')


def prepare_annotations(stage_root, dataset_names):
    records = []
    for dataset_name in dataset_names:
        spec = DATASETS[dataset_name]
        for name in spec['annotations']:
            source = stage_root / 'annotations' / 'NOD' / name
            target = stage_root / 'annotations' / 'NOD_2x' / name
            payload = scale_annotation(source, target)
            records.append(dict(
                dataset=dataset_name,
                source=str(source),
                source_sha256=sha256(source),
                target=str(target),
                target_sha256=sha256(target),
                images=len(payload['images']),
                annotations=len(payload['annotations']),
            ))
    return records


def build_tasks(stage_root, dataset_names, overwrite):
    import numpy as np

    tasks = []
    for dataset_name in dataset_names:
        spec = DATASETS[dataset_name]
        source_dir = stage_root / spec['source']
        for name in spec['annotations']:
            annotation_path = stage_root / 'annotations' / 'NOD_2x' / name
            payload = json.loads(annotation_path.read_text())
            split = annotation_split(annotation_path)
            destination = stage_root / spec['destination'](split)
            destination.mkdir(parents=True, exist_ok=True)
            for image in payload['images']:
                stem = Path(image['file_name']).stem
                source = source_dir / f"{stem}{spec['suffix']}"
                target = destination / image['file_name']
                expected = (int(image['height']), int(image['width']), 4)
                if not source.is_file():
                    raise FileNotFoundError(source)
                if target.is_file() and not overwrite:
                    try:
                        array = np.load(target, mmap_mode='r')
                        if tuple(array.shape) == expected:
                            continue
                    except Exception:
                        pass
                tasks.append(dict(
                    dataset=dataset_name,
                    source=str(source),
                    target=str(target),
                    height=expected[0],
                    width=expected[1],
                ))
    return tasks


def convert_one(task):
    import cv2
    import numpy as np
    import rawpy

    source = Path(task['source'])
    target = Path(task['target'])
    with rawpy.imread(str(source)) as raw:
        mosaic = raw.raw_image
        array = np.stack((
            mosaic[0::2, 0::2], mosaic[0::2, 1::2],
            mosaic[1::2, 0::2], mosaic[1::2, 1::2]), axis=-1)
    array = cv2.resize(
        array, (task['width'], task['height']),
        interpolation=cv2.INTER_LINEAR)
    expected = (task['height'], task['width'], 4)
    if tuple(array.shape) != expected:
        raise ValueError(
            f'{source} produced {array.shape}, expected {expected}')
    temporary = target.with_name(f'.{target.name}.{os.getpid()}.tmp')
    with temporary.open('wb') as handle:
        np.save(handle, array)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return str(target)


def main():
    args = parse_args()
    stage_root = args.stage_root.resolve()
    annotation_records = prepare_annotations(stage_root, args.dataset)
    tasks = build_tasks(stage_root, args.dataset, args.overwrite)
    if args.limit is not None:
        tasks = tasks[:args.limit]
    print(f'stage_root={stage_root}', flush=True)
    print(f"datasets={','.join(args.dataset)}", flush=True)
    print(f'pending_files={len(tasks)}', flush=True)

    completed = 0
    with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp.get_context('spawn')) as executor:
        futures = [executor.submit(convert_one, task) for task in tasks]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed == 1 or completed % 100 == 0 or completed == len(tasks):
                print(f'completed={completed}/{len(tasks)}', flush=True)

    audit = dict(
        schema_version=1,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        scale=SCALE,
        datasets=args.dataset,
        workers=args.workers,
        converted_this_run=completed,
        annotation_records=annotation_records,
        converter_sha256=sha256(Path(__file__)),
    )
    audit_name = 'nod_2x_conversion_{}.json'.format(
        '_'.join(name.replace('-', '_') for name in args.dataset))
    atomic_json(stage_root / 'audit' / audit_name, audit)


if __name__ == '__main__':
    main()
