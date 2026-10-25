#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


OUT_HEIGHT = 928
OUT_WIDTH = 1440
COORD_SCALE = 0.5
SMOKE_SEED = 3407
SMOKE_QUOTAS = {
    'train': {'day': 25, 'night': 75},
    'val': {'day': 20, 'night': 20},
}
CLASS_NAMES = ('Tram', 'Car', 'Truck', 'Cyclist', 'Pedestrian')
CATEGORY_ID_BY_NAME = {name: idx + 1 for idx, name in enumerate(CLASS_NAMES)}
BOX_COLORS = {
    'Tram': (255, 64, 64),
    'Car': (80, 176, 255),
    'Truck': (255, 180, 0),
    'Cyclist': (96, 220, 96),
    'Pedestrian': (255, 96, 208),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build COCO annotations for half-resolution ROD RGB NPY images.')
    parser.add_argument(
        '--train-label-dir',
        type=Path,
        required=True,
        help='Directory containing training LabelMe json files.')
    parser.add_argument(
        '--val-label-dir',
        type=Path,
        required=True,
        help='Directory containing validation txt annotations.')
    parser.add_argument(
        '--output-root',
        type=Path,
        required=True,
        help='Root directory for generated smoke/full datasets.')
    parser.add_argument(
        '--variant',
        choices=('smoke', 'full', 'both'),
        default='full',
        help='Which dataset variant(s) to export.')
    parser.add_argument(
        '--qc-count',
        type=int,
        default=0,
        help='How many smoke overlay images to render in total.')
    return parser.parse_args()


def gather_stems(label_dir: Path, suffix: str) -> list[str]:
    stems = sorted(path.stem for path in label_dir.glob(f'*.{suffix}'))
    if not stems:
        raise FileNotFoundError(f'No *.{suffix} files found under {label_dir}')
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


def load_subset_stems(dataset_root: Path, split_name: str, fallback_stems: list[str]) -> list[str]:
    metadata_path = dataset_root / 'metadata' / 'splits.json'
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        return metadata['splits'][split_name]['stems']
    if dataset_root.name == 'smoke':
        return select_smoke_subset(fallback_stems, split_name)
    return fallback_stems


def scale_clip_bbox(x1: float, y1: float, x2: float, y2: float) -> tuple[list[float], float] | None:
    x1, x2 = sorted((float(x1), float(x2)))
    y1, y2 = sorted((float(y1), float(y2)))
    x1 *= COORD_SCALE
    x2 *= COORD_SCALE
    y1 *= COORD_SCALE
    y2 *= COORD_SCALE

    x1 = min(max(x1, 0.0), float(OUT_WIDTH))
    x2 = min(max(x2, 0.0), float(OUT_WIDTH))
    y1 = min(max(y1, 0.0), float(OUT_HEIGHT))
    y2 = min(max(y2, 0.0), float(OUT_HEIGHT))
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None
    bbox = [round(x1, 3), round(y1, 3), round(width, 3), round(height, 3)]
    return bbox, round(width * height, 3)


def parse_train_boxes(json_path: Path) -> list[tuple[str, list[float], float]]:
    data = json.loads(json_path.read_text(encoding='utf-8'))
    parsed: list[tuple[str, list[float], float]] = []
    for shape in data.get('shapes', []):
        label = shape.get('label')
        if label not in CATEGORY_ID_BY_NAME:
            continue
        if shape.get('shape_type', 'rectangle') != 'rectangle':
            continue
        points = shape.get('points', [])
        if len(points) < 2:
            continue
        scaled = scale_clip_bbox(points[0][0], points[0][1], points[1][0], points[1][1])
        if scaled is None:
            continue
        bbox, area = scaled
        parsed.append((label, bbox, area))
    return parsed


def parse_val_boxes(txt_path: Path) -> list[tuple[str, list[float], float]]:
    parsed: list[tuple[str, list[float], float]] = []
    for line in txt_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        label, x1, y1, x2, y2 = parts
        if label not in CATEGORY_ID_BY_NAME:
            continue
        scaled = scale_clip_bbox(float(x1), float(y1), float(x2), float(y2))
        if scaled is None:
            continue
        bbox, area = scaled
        parsed.append((label, bbox, area))
    return parsed


def categories_block() -> list[dict]:
    return [{'id': CATEGORY_ID_BY_NAME[name], 'name': name} for name in CLASS_NAMES]


def build_coco_dataset(
    stems: list[str],
    label_dir: Path,
    parser_kind: str,
) -> tuple[dict, dict[str, int]]:
    images = []
    annotations = []
    ann_id = 1
    empty_images = 0
    box_count = 0

    for image_id, stem in enumerate(stems, start=1):
        images.append({
            'id': image_id,
            'file_name': f'{stem}.npy',
            'width': OUT_WIDTH,
            'height': OUT_HEIGHT,
        })
        if parser_kind == 'train':
            boxes = parse_train_boxes(label_dir / f'{stem}.json')
        elif parser_kind == 'val':
            boxes = parse_val_boxes(label_dir / f'{stem}.txt')
        else:
            raise KeyError(f'Unsupported parser kind: {parser_kind}')

        if not boxes:
            empty_images += 1

        for label, bbox, area in boxes:
            annotations.append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': CATEGORY_ID_BY_NAME[label],
                'bbox': bbox,
                'area': area,
                'iscrowd': 0,
            })
            ann_id += 1
            box_count += 1

    coco = {
        'info': {
            'description': 'ROD packed RGB object detection dataset',
        },
        'licenses': [],
        'images': images,
        'annotations': annotations,
        'categories': categories_block(),
    }
    stats = {
        'images': len(images),
        'annotations': box_count,
        'empty_images': empty_images,
    }
    return coco, stats


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def load_annotations_by_image(coco: dict) -> dict[int, list[dict]]:
    ann_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco['annotations']:
        ann_by_image[ann['image_id']].append(ann)
    return ann_by_image


def render_image_for_qc(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    upper = float(np.percentile(img, 99.5))
    if upper <= 0:
        upper = float(img.max()) if img.size else 1.0
    if upper <= 0:
        upper = 1.0
    img = np.clip(img / upper, 0.0, 1.0)
    img = np.power(img, 1.0 / 2.2)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def save_qc_overlays(
    dataset_root: Path,
    train_coco: dict,
    val_coco: dict,
    qc_count: int,
) -> None:
    if qc_count <= 0:
        return
    from PIL import Image, ImageDraw

    qc_dir = dataset_root.parent / 'qc' / f'{dataset_root.name}_overlay'
    qc_dir.mkdir(parents=True, exist_ok=True)

    selections = [
        ('train', dataset_root / 'images' / 'train', train_coco, max(1, qc_count // 2)),
        ('val', dataset_root / 'images' / 'val', val_coco, qc_count - max(1, qc_count // 2)),
    ]
    for split_name, image_dir, coco, split_qc_count in selections:
        if split_qc_count <= 0:
            continue
        ann_by_image = load_annotations_by_image(coco)
        eligible = [image for image in coco['images'] if ann_by_image.get(image['id'])]
        for image in eligible[:split_qc_count]:
            npy_path = image_dir / image['file_name']
            img = np.load(npy_path)
            vis = render_image_for_qc(img)
            canvas = Image.fromarray(vis, mode='RGB')
            draw = ImageDraw.Draw(canvas)
            for ann in ann_by_image[image['id']]:
                x, y, w, h = ann['bbox']
                label = CLASS_NAMES[ann['category_id'] - 1]
                color = BOX_COLORS[label]
                draw.rectangle((x, y, x + w, y + h), outline=color, width=2)
                draw.text((x + 2, max(0, y - 12)), label, fill=color)
            canvas.save(qc_dir / f'{split_name}_{Path(image["file_name"]).stem}.png')


def export_variant(
    dataset_root: Path,
    train_label_dir: Path,
    val_label_dir: Path,
    qc_count: int,
    fallback_train_stems: list[str],
    fallback_val_stems: list[str],
) -> dict[str, dict[str, int]]:
    train_stems = load_subset_stems(dataset_root, 'train', fallback_train_stems)
    val_stems = load_subset_stems(dataset_root, 'val', fallback_val_stems)

    train_coco, train_stats = build_coco_dataset(train_stems, train_label_dir, 'train')
    val_coco, val_stats = build_coco_dataset(val_stems, val_label_dir, 'val')
    train_day_stems = [stem for stem in train_stems if stem.startswith('day-')]
    train_night_stems = [stem for stem in train_stems if stem.startswith('night-')]
    val_day_stems = [stem for stem in val_stems if stem.startswith('day-')]
    val_night_stems = [stem for stem in val_stems if stem.startswith('night-')]
    train_day_coco, train_day_stats = build_coco_dataset(train_day_stems, train_label_dir, 'train')
    train_night_coco, train_night_stats = build_coco_dataset(train_night_stems, train_label_dir, 'train')
    val_day_coco, val_day_stats = build_coco_dataset(val_day_stems, val_label_dir, 'val')
    val_night_coco, val_night_stats = build_coco_dataset(val_night_stems, val_label_dir, 'val')

    annotations_dir = dataset_root / 'annotations'
    save_json(annotations_dir / 'train.json', train_coco)
    save_json(annotations_dir / 'train_day.json', train_day_coco)
    save_json(annotations_dir / 'train_night.json', train_night_coco)
    save_json(annotations_dir / 'val.json', val_coco)
    save_json(annotations_dir / 'val_day.json', val_day_coco)
    save_json(annotations_dir / 'val_night.json', val_night_coco)

    save_qc_overlays(dataset_root, train_coco, val_coco, qc_count)

    return {
        'train': train_stats,
        'train_day': train_day_stats,
        'train_night': train_night_stats,
        'val': val_stats,
        'val_day': val_day_stats,
        'val_night': val_night_stats,
    }


def main() -> None:
    args = parse_args()
    train_stems_all = gather_stems(args.train_label_dir, 'json')
    val_stems_all = gather_stems(args.val_label_dir, 'txt')

    variants: list[str] = []
    if args.variant in ('smoke', 'both'):
        variants.append('smoke')
    if args.variant in ('full', 'both'):
        variants.append('full')

    summary: dict[str, dict[str, dict[str, int]]] = {}
    for variant_name in variants:
        dataset_root = args.output_root / variant_name
        summary[variant_name] = export_variant(
            dataset_root=dataset_root,
            train_label_dir=args.train_label_dir,
            val_label_dir=args.val_label_dir,
            qc_count=args.qc_count if variant_name == 'smoke' else 0,
            fallback_train_stems=train_stems_all,
            fallback_val_stems=val_stems_all,
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
