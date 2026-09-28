# RAWild

Official implementation of **RAWild: Sensor-Agnostic RAW Object Detection via Physics-Guided Curve and Grid Modeling**.

**[Project Page](https://i2wm.github.io/RAWild/)**

RAWild adds a lightweight RAW adapter before the downstream vision backbone. The adapter predicts a per-image Bezier tone curve and a bilateral-grid color transform, enabling robust perception across RAW images from different sensors, exposure levels, bit depths, and spectral responses.

<p align="center">
  <img src="picture/pipeline.png" width="900">
</p>

This release includes object detection, semantic segmentation, RAW simulation, and adapter visualization code.

## Assets

Datasets and checkpoints will be released separately.

- Dataset download: 
  - RAWild_Mutiraw: [Google Drive](https://drive.google.com/drive/folders/1zmKUoUyjMoSE2jWIehoAxDQlutD6LC0a?usp=sharing)
  - Simulation: [Google Drive](https://drive.google.com/drive/folders/1-x90kmXguv7vsjsIs9w3Kp459byy_hhN?usp=sharing)

Dataset folders include `RAWild_Mutiraw` and `Simulation` under the Drive dataset folder.
- Checkpoint download: [Google Drive](https://drive.google.com/drive/folders/1gJlPLMO7Ty0XszLQTJkoZUIxeqbOwKqd?usp=sharing)

Checkpoint folders are organized as `Checkpoints/Det` for object detection and `Checkpoints/Seg` for semantic segmentation.
Detection checkpoint files are named as `<Backbone>-<Dataset>.pth`, for example `ResNet50-PAS.NM.pth`; segmentation checkpoint files are named as `<Backbone>-<Domain>.pth`, for example `MiT-B3-normal.pth`.

Set paths before running the examples:

```bash
export RAWILD_DATA_ROOT=<RAWild_datasets>
export RAWILD_PASCAL_NPY_ROOT="$RAWILD_DATA_ROOT/PASCAL_RAW_npy"
export RAWILD_PRETRAINED_ROOT=<checkpoints/pretrained>
export RAWILD_RELEASE_CKPT_ROOT=<checkpoints/released>
export RAWILD_WORK_DIR_ROOT=work_dirs

export RAWILD_MMSEG_DATA_ROOT="$RAWILD_DATA_ROOT/ADEChallengeData2016"
export RAWILD_MMSEG_WORK_DIR_ROOT=work_dirs/rawild_mmseg
```

`RAWILD_DATA_ROOT` should contain `PASCAL_RAW`, `PASCAL_RAW_npy`, `LOD_BMVC2021`, `ROD_dataset`, `AODRaw_dataset`, and `ADEChallengeData2016`.

## Installation

Detection:

```bash
conda create -n RAWild_mmdet python=3.8 -y
conda activate RAWild_mmdet

cd mmdetection_github
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install -r requirements.txt
pip install -v -e .
```

Segmentation:

```bash
conda create -n RAWild_mmseg python=3.8 -y
conda activate RAWild_mmseg

cd mmsegmentation_github
pip install -r requirements.txt
pip install -e . --no-deps
```

The segmentation dependency snapshot was verified with Python 3.8, PyTorch 2.1.0 + CUDA 12.1, MMCV 2.1.0, and MMEngine 0.10.4.

## Object Detection

Configs:

| Setting | ResNet-50 | Swin-T |
| --- | --- | --- |
| PASCALRAW normal | `config/RAWild_resnet50/pas_nm.py` | `config/RAWild_swint/pas_nm.py` |
| PASCALRAW low-light | `config/RAWild_resnet50/pas_low.py` | `config/RAWild_swint/pas_low.py` |
| PASCALRAW over-exposure | `config/RAWild_resnet50/pas_oe.py` | `config/RAWild_swint/pas_oe.py` |
| LOD | `config/RAWild_resnet50/lod.py` | `config/RAWild_swint/lod.py` |
| PASCALRAW + LOD mix | `config/RAWild_resnet50/pascal_lod_mix.py` | `config/RAWild_swint/pascal_lod_mix.py` |
| ROD | `config/RAWild_resnet50/rod.py` | `config/RAWild_swint/rod.py` |
| AODRaw | `config/RAWild_resnet50/aodraw.py` | `config/RAWild_swint/aodraw.py` |

Reported results of RAWild:

| Backbone | PAS.LOW @50/@75 | PAS.NM @50/@75 | PAS.OE @50/@75 | LOD @50/@75 | ROD @50/@75 | AODRAW @50/@75 |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet-50 | 0.8911 / 0.7332 | 0.9019 / 0.7690 | 0.9020 / 0.7690 | 0.6909 / 0.4363 | 0.5541 / 0.3752 | 0.3623 / 0.2311 |
| Swin-T | 0.9083 / 0.7406 | 0.9342 / 0.7792 | 0.9313 / 0.7921 | 0.6986 / 0.5057 | 0.5191 / 0.3467 | 0.4394 / 0.3263 |

Train:

```bash
cd mmdetection_github

python tools/train.py config/RAWild_resnet50/pascal_lod_mix.py
python tools/train.py config/RAWild_swint/pascal_lod_mix.py
```

Distributed training:

```bash
bash tools/dist_train.sh config/RAWild_resnet50/pascal_lod_mix.py 4
bash tools/dist_train.sh config/RAWild_swint/pascal_lod_mix.py 4
```

Evaluate:

```bash
python tools/test.py \
  config/RAWild_resnet50/pas_nm.py \
  "$RAWILD_RELEASE_CKPT_ROOT/ResNet50-PAS.NM.pth"

python tools/test.py \
  config/RAWild_swint/pas_nm.py \
  "$RAWILD_RELEASE_CKPT_ROOT/SwinT-PAS.NM.pth"
```

Use the config table above to switch datasets or backbones.

## Semantic Segmentation

Configs:

| Backbone | Normal | Low | Over Exposure | Mix |
| --- | --- | --- | --- | --- |
| MiT-B0 | `config/RAWild_mitb0/normal.py` | `config/RAWild_mitb0/low.py` | `config/RAWild_mitb0/oe.py` | `config/RAWild_mitb0/mix.py` |
| MiT-B3 | `config/RAWild_mitb3/normal.py` | `config/RAWild_mitb3/low.py` | `config/RAWild_mitb3/oe.py` | `config/RAWild_mitb3/mix.py` |
| MiT-B5 | `config/RAWild_mitb5/normal.py` | `config/RAWild_mitb5/low.py` | `config/RAWild_mitb5/oe.py` | `config/RAWild_mitb5/mix.py` |

Reported results of RAWild:

| Backbone | LOW mIoU | NM mIoU | OE mIoU |
| --- | --- | --- | --- |
| MiT-B0 | 0.2872 | 0.3534 | 0.3372 |
| MiT-B3 | 0.3957 | 0.4516 | 0.4381 |
| MiT-B5 | 0.4082 | 0.4708 | 0.4560 |

Train:

```bash
cd mmsegmentation_github

python tools/train.py config/RAWild_mitb0/mix.py
python tools/train.py config/RAWild_mitb3/mix.py
python tools/train.py config/RAWild_mitb5/mix.py
```

Distributed training:

```bash
bash tools/dist_train.sh config/RAWild_mitb0/mix.py 4
bash tools/dist_train.sh config/RAWild_mitb3/mix.py 4
bash tools/dist_train.sh config/RAWild_mitb5/mix.py 4
```

Evaluate:

```bash
python tools/test.py \
  config/RAWild_mitb0/mix.py \
  "$RAWILD_RELEASE_CKPT_ROOT/rawild_mitb0_mix.pth"

python tools/test.py \
  config/RAWild_mitb3/mix.py \
  "$RAWILD_RELEASE_CKPT_ROOT/rawild_mitb3_mix.pth"

python tools/test.py \
  config/RAWild_mitb5/mix.py \
  "$RAWILD_RELEASE_CKPT_ROOT/rawild_mitb5_mix.pth"
```

For paper-style evaluation, use the single-scale sliding-window path in the config and do not enable TTA.

## RAW Simulation

The released dataset already includes processed PASCALRAW `.npy` files under `PASCAL_RAW_npy`. If you start from original PASCAL RAW `.nef` files, preprocess them with:

```bash
cd mmdetection_github

python PASCAL_RAW_pre_process.py \
  --raw-root <PASCAL_RAW_NEF_DIR> \
  --out-root "$RAWILD_DATA_ROOT/PASCAL_RAW/original"
```

Generate synthetic spectrum and mixed-bit PASCALRAW data:

```bash
python tools/syn_spec/generate_pascal_normal_mixbit.py \
  --source-root "$RAWILD_PASCAL_NPY_ROOT/demosaic_normal_12bit" \
  --train-list "$RAWILD_DATA_ROOT/PASCAL_RAW/trainval/train.txt" \
  --val-list "$RAWILD_DATA_ROOT/PASCAL_RAW/trainval/val.txt" \
  --output-root "$RAWILD_WORK_DIR_ROOT/syn_spec/pascal_normal_mixbit" \
  --bit-depths 8 9 10 11 12
```

Visualize camera-seed variants:

```bash
python tools/syn_spec/visualize_camera_seed_views.py \
  --sample-id 2014_000022 \
  --source-root "$RAWILD_PASCAL_NPY_ROOT/demosaic_normal_12bit" \
  --output-root "$RAWILD_WORK_DIR_ROOT/syn_spec/camera_seed_views"
```

## Visualization

| Input | Bezier |
| --- | --- |
| <img src="picture/official_rgb_original.png" width="360"> | <img src="picture/bezier_only.png" width="360"> |

| Bezier + Grid | Curve |
| --- | --- |
| <img src="picture/bezier_grid.png" width="360"> | <img src="picture/bezier_curve.png" width="360"> |

Generate adapter visualizations:

```bash
cd mmdetection_github

NO_ALBUMENTATIONS_UPDATE=1 python tools/visual_selection/visualize_bezier.py \
  --config config/RAWild_resnet50/pas_nm.py \
  --checkpoint "$RAWILD_RELEASE_CKPT_ROOT/ResNet50-PAS.NM.pth" \
  --image-key 2014_000001 \
  --out-dir "$RAWILD_WORK_DIR_ROOT/visual_selection/pas_nm"
```

Export the Bezier curve only:

```bash
NO_ALBUMENTATIONS_UPDATE=1 python tools/visual_selection/visualize_bezier.py \
  --config config/RAWild_resnet50/pas_nm.py \
  --checkpoint "$RAWILD_RELEASE_CKPT_ROOT/ResNet50-PAS.NM.pth" \
  --image-key 2014_000001 \
  --out-dir "$RAWILD_WORK_DIR_ROOT/visual_selection/pas_nm_curve" \
  --curve-only
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{liu2026rawild,
  title={RAWild: Sensor-Agnostic RAW Object Detection via Physics-Guided Curve and Grid Modeling},
  author={Liu, Shuhong and Chang, Gengjia and Liu, Jun and Chu, Xuangeng and Zheng, Yinqiang and Harada, Tatsuya and Cui, Ziteng},
  journal={arXiv preprint arXiv:2605.05941},
  year={2026}
}
```
