# RAWild Semantic Segmentation

This repository contains the RAWild semantic segmentation code path for ADE20K-RAW. The public release is intentionally trimmed to keep only the RAWild method family and the runtime modules required to train and evaluate it.

Use the top-level `../README.md` for the full release instructions, including dataset/checkpoint placeholders and end-to-end commands.

## Method

The segmentation backbone is `RAW_BilateralGrid_MixVisionTransformer_DIA`, a MiT/SegFormer backbone with a DIA bilateral-grid RAW adapter. The released configs keep the RAWild mainline settings:

- `grid_depth=8`
- `transformer_dim=128`
- `curve_n=6`
- `use_bezier=True`
- `bezier_type='init_bias'`
- `dia_k=0.05`
- `dia_activation='exp_tanh'`
- `hist_bins=64` soft quantile conditioning
- `cond_injection_type='adaln'`

## Configs

Public configs live under `config/`:

| Backbone | Normal | Low | Over Exposure | Mix |
| --- | --- | --- | --- | --- |
| MiT-B0 | `config/RAWild_mitb0/normal.py` | `config/RAWild_mitb0/low.py` | `config/RAWild_mitb0/oe.py` | `config/RAWild_mitb0/mix.py` |
| MiT-B3 | `config/RAWild_mitb3/normal.py` | `config/RAWild_mitb3/low.py` | `config/RAWild_mitb3/oe.py` | `config/RAWild_mitb3/mix.py` |
| MiT-B5 | `config/RAWild_mitb5/normal.py` | `config/RAWild_mitb5/low.py` | `config/RAWild_mitb5/oe.py` | `config/RAWild_mitb5/mix.py` |

The `mix` recipe trains on three ADE20K-RAW image branches and validates on low-light RAW:

- `images/training_raw_low`
- `images/training_raw_over_exp`
- `images/training_raw`
- validation: `images/validation_raw_low`

Use the `normal`, `low`, and `oe` configs to evaluate the LOW / NM / OE exposure conditions reported in the paper. The default test path is single-scale sliding-window inference; do not pass `--tta` for paper-style evaluation.

## Dataset Layout

By default the configs expect the dataset at:

```text
data/ADE20K/ADEChallengeData2016
```

Expected structure:

```text
ADEChallengeData2016/
  images/
    training_raw/
    training_raw_low/
    training_raw_over_exp/
    validation_raw/
    validation_raw_low/
    validation_raw_over_exp/
  annotations/
    training/
    validation/
```

If your dataset is stored elsewhere, set:

```bash
export RAWILD_MMSEG_DATA_ROOT=<ADEChallengeData2016>
export RAWILD_MMSEG_WORK_DIR_ROOT=work_dirs/rawild_mmseg
export RAWILD_RELEASE_CKPT_ROOT=<checkpoints/released>
```

Do not commit machine-specific absolute paths.

## Environment

A verified CUDA 12.1 / PyTorch 2.1 dependency snapshot is provided:

```bash
conda create -n RAWild_mmseg python=3.8 -y
conda activate RAWild_mmseg
pip install -r requirements.txt
pip install -e . --no-deps
```

The verified environment used:

- Python 3.8
- PyTorch 2.1.0 + CUDA 12.1
- torchvision 0.16.0
- MMCV 2.1.0
- MMEngine 0.10.4
- MMSegmentation 1.2.1-compatible codebase

## Training

The released paper recipe uses 200k iterations, 1.5k linear warmup, polynomial decay, AdamW with base learning rate `6e-5`, and 10x learning-rate multipliers for the RAWild adapter and decode head. Public configs keep `batch_size=4` per GPU; use 4 GPUs for global batch size 16, or override the batch size at runtime for a single-GPU reproduction.

Single-GPU example:

```bash
python tools/train.py config/RAWild_mitb0/mix.py
```

Distributed example:

```bash
bash tools/dist_train.sh config/RAWild_mitb0/mix.py 4
```

## Evaluation

```bash
python tools/test.py config/RAWild_mitb0/mix.py "$RAWILD_RELEASE_CKPT_ROOT/rawild_mitb0_mix.pth"
```

Use `RAWILD_RELEASE_CKPT_ROOT` for public checkpoints. Private checkpoint paths should stay outside committed files.

## Release Scope

This release has removed baseline method families and original RAW-Adapter comparison configs. It keeps only the RAWild segmentation method and the minimal source/config layer needed by the public RAWild configs.
