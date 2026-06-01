# Visual Selection

This directory keeps one visualization entrypoint for RAWild adapter diagnostics.

## Usage

```bash
NO_ALBUMENTATIONS_UPDATE=1 python \
  tools/visual_selection/visualize_bezier.py \
  --config config/RAWild_resnet50/pas_nm.py \
  --checkpoint checkpoints/model.pth \
  --image-key 2014_000001 \
  --out-dir work_dirs/rawild_visual
```

Outputs:

- `linear_input.png`: loaded linear RAW input branch.
- `bezier_only.png`: input after the learned Bezier curve only.
- `bezier_grid.png`: full adapter output, Bezier plus grid when enabled.
- `bezier_curve.png`: pure Bezier response curve.
- `metadata.json`: checkpoint/config/image metadata and control points.

The image must be reachable from the selected config dataloader. Use
`--image-index` instead of `--image-key` when visualizing by dataset index.
