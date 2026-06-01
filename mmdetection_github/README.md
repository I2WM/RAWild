# RAWild Object Detection

This directory contains the MMDetection code path for RAWild object detection.

Recommended environment name: `RAWild_mmdet`.

Use the top-level README for installation, dataset layout, checkpoint placeholders, training commands, evaluation commands, RAW simulation, and visualization:

```text
../README.md
```

Main config roots:

```text
config/RAWild_resnet50
config/RAWild_swint
```

Primary examples:

```bash
python tools/train.py config/RAWild_resnet50/pas_nm.py
python tools/test.py config/RAWild_resnet50/pas_nm.py "$RAWILD_RELEASE_CKPT_ROOT/rawild_resnet50_pas_nm.pth"

python tools/train.py config/RAWild_swint/pas_nm.py
python tools/test.py config/RAWild_swint/pas_nm.py "$RAWILD_RELEASE_CKPT_ROOT/rawild_swint_pas_nm.pth"
```

Detection path variables:

```bash
export RAWILD_DATA_ROOT=<RAWild_datasets>
export RAWILD_PASCAL_NPY_ROOT="$RAWILD_DATA_ROOT/PASCAL_RAW_npy"
export RAWILD_PRETRAINED_ROOT=<checkpoints/pretrained>
export RAWILD_RELEASE_CKPT_ROOT=<checkpoints/released>
export RAWILD_WORK_DIR_ROOT=work_dirs
```
