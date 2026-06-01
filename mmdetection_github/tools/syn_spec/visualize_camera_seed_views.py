#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SYN_SPEC_DIR = Path(__file__).resolve().parent
if str(SYN_SPEC_DIR) not in sys.path:
    sys.path.insert(0, str(SYN_SPEC_DIR))

from syn_spec import RAWSimulator


DEFAULT_SOURCE_ROOT = Path("data/cgj_simulation/demosaic_normal_12bit")
DEFAULT_OUTPUT_ROOT = Path("work_dirs/syn_spec/camera_seed_views")
REPRESENTATIVE_SEEDS = [6, 59, 76, 118, 222]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render camera-seed previews with explicit RAW/viewer semantics."
    )
    parser.add_argument("--sample-id", default="2014_000022")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bit-depth", type=int, default=12)
    parser.add_argument("--seed-count", type=int, default=256)
    parser.add_argument("--first-n", type=int, default=24)
    parser.add_argument(
        "--representative-seeds", type=int, nargs="+", default=REPRESENTATIVE_SEEDS
    )
    return parser.parse_args()


def clip_gamma(arr):
    return np.clip(arr, 0.0, 1.0) ** (1.0 / 2.2)


def save_rgb(path, image):
    plt.imsave(path, np.clip(image, 0.0, 1.0))


def classify_seed(r_over_g, b_over_g):
    if abs(r_over_g - 1.0) <= 0.04 and abs(b_over_g - 1.0) <= 0.04:
        return "neutral"
    if r_over_g >= 1.03 and b_over_g >= 1.03:
        return "magenta/pink"
    if r_over_g <= 0.97 and b_over_g <= 0.97:
        return "green"
    if b_over_g >= 1.08 and r_over_g >= 0.95:
        return "blue"
    if b_over_g >= 1.03:
        return "cyan/blue"
    if r_over_g >= 1.03 and b_over_g <= 0.97:
        return "warm/yellow"
    return "other"


CLASS_COLORS = {
    "neutral": "#4C78A8",
    "magenta/pink": "#D81B60",
    "green": "#2CA02C",
    "blue": "#1F77B4",
    "cyan/blue": "#17BECF",
    "warm/yellow": "#E69F00",
    "other": "#7F7F7F",
}


def compute_wb_gain(raw):
    ch_means = raw.mean(axis=(0, 1)) + 1e-8
    return ch_means[1] / ch_means


def quantize(arr, bit_depth):
    levels = 2 ** int(bit_depth) - 1
    return np.floor(np.clip(arr, 0.0, 1.0) * levels + 0.5) / levels


def simulate_camera_seed(simulator, raw, wb_gain, camera_seed, bit_depth):
    wl = simulator.wavelengths
    raw_balanced = raw * wb_gain[None, None, :]
    illuminant = simulator._planckian_spectrum(6500.0, wl)
    tint_gain = simulator._tint_to_diagonal(illuminant, 0.0)
    rng = np.random.default_rng(camera_seed)
    target_camera = simulator._sample_camera(rng)
    matrix = np.diag(tint_gain) @ simulator._cross_camera_matrix(target_camera, illuminant)
    flat = raw_balanced.reshape(-1, 3)
    out = (matrix @ flat.T).T.reshape(raw.shape)
    out = out / wb_gain[None, None, :]
    out = np.tanh(out)
    out = quantize(out, bit_depth)
    mean_rgb = out.mean(axis=(0, 1))
    return out, {
        "camera_seed": int(camera_seed),
        "mean_rgb": [float(v) for v in mean_rgb],
        "r_over_g": float(mean_rgb[0] / (mean_rgb[1] + 1e-8)),
        "b_over_g": float(mean_rgb[2] / (mean_rgb[1] + 1e-8)),
    }


def simulate_mean_camera(simulator, raw, wb_gain, bit_depth):
    wl = simulator.wavelengths
    raw_balanced = raw * wb_gain[None, None, :]
    illuminant = simulator._planckian_spectrum(6500.0, wl)
    tint_gain = simulator._tint_to_diagonal(illuminant, 0.0)
    target_camera = np.zeros((len(wl), 3), dtype=np.float32)
    for channel_idx, channel_name in enumerate(["R", "G", "B"]):
        stats = simulator.stats[channel_name]
        sigma = stats["mean"]
        curve = simulator.pca[channel_name].inverse_transform(sigma[None])[0]
        curve = np.clip(curve, 0.0, None)
        curve /= curve.max() + 1e-8
        target_camera[:, channel_idx] = curve
    matrix = np.diag(tint_gain) @ simulator._cross_camera_matrix(target_camera, illuminant)
    flat = raw_balanced.reshape(-1, 3)
    out = (matrix @ flat.T).T.reshape(raw.shape)
    out = out / wb_gain[None, None, :]
    out = np.tanh(out)
    out = quantize(out, bit_depth)
    mean_rgb = out.mean(axis=(0, 1))
    return out, [float(v) for v in mean_rgb]


def make_raw_preview(arr, scalar):
    return clip_gamma(arr / (scalar + 1e-8))


def make_gray_world_preview(arr):
    mean_rgb = arr.mean(axis=(0, 1)) + 1e-8
    display_gain = mean_rgb.mean() / mean_rgb
    lifted = np.clip(arr * display_gain[None, None, :], 0.0, None)
    scalar = float(np.percentile(lifted, 99.5))
    return clip_gamma(lifted / (scalar + 1e-8)), display_gain, scalar


def make_gain_preview(arr, gain_rgb):
    lifted = np.clip(arr * gain_rgb[None, None, :], 0.0, None)
    scalar = float(np.percentile(lifted, 99.5))
    return clip_gamma(lifted / (scalar + 1e-8)), scalar


def render_first_n_grid(path, sample_id, mode_name, seeds, previews, stats, cols=6):
    rows = int(np.ceil(len(seeds) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.1, rows * 2.55), dpi=200)
    axes = np.asarray(axes).reshape(rows, cols)
    for axis in axes.ravel():
        axis.axis("off")
    for idx, seed in enumerate(seeds):
        axis = axes[idx // cols, idx % cols]
        axis.imshow(previews[seed])
        axis.axis("off")
        info = stats[seed]
        axis.set_title(
            (
                f"camera seed {seed}\n"
                f"{info['class']} | R/G={info['r_over_g']:.3f} | B/G={info['b_over_g']:.3f}"
            ),
            fontsize=8,
            pad=5,
        )
    fig.suptitle(
        f"{sample_id} | first {len(seeds)} camera seeds | {mode_name}",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def render_representative_grid(path, sample_id, seeds, raw_previews, viewer_previews, stats):
    fig, axes = plt.subplots(2, len(seeds), figsize=(len(seeds) * 3.1, 5.8), dpi=220)
    row_titles = ["RAW-style preview", "Viewer preview"]
    for row_idx, row_name in enumerate(row_titles):
        for col_idx, seed in enumerate(seeds):
            axis = axes[row_idx, col_idx]
            axis.imshow(raw_previews[seed] if row_idx == 0 else viewer_previews[seed])
            axis.axis("off")
            info = stats[seed]
            title = (
                f"camera seed {seed}\n"
                f"{info['class']} | R/G={info['r_over_g']:.3f} | B/G={info['b_over_g']:.3f}"
            )
            if row_idx == 0:
                axis.set_title(title, fontsize=9, pad=5)
            if col_idx == 0:
                axis.text(
                    -0.08,
                    0.5,
                    row_name,
                    rotation=90,
                    va="center",
                    ha="right",
                    transform=axis.transAxes,
                    fontsize=10,
                )
    fig.suptitle(
        f"{sample_id} | representative camera-seed contrasts",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0.015, 0, 1, 0.97))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def render_scatter(path, sample_id, stats, highlighted):
    fig, axis = plt.subplots(figsize=(7.2, 6.6), dpi=220)
    for seed, info in stats.items():
        axis.scatter(
            info["r_over_g"],
            info["b_over_g"],
            s=26,
            c=CLASS_COLORS[info["class"]],
            alpha=0.7,
            edgecolors="none",
        )
    for seed in highlighted:
        info = stats[seed]
        axis.scatter(
            info["r_over_g"],
            info["b_over_g"],
            s=92,
            c=CLASS_COLORS[info["class"]],
            edgecolors="black",
            linewidths=0.7,
        )
        axis.text(
            info["r_over_g"] + 0.003,
            info["b_over_g"] + 0.003,
            str(seed),
            fontsize=8,
        )
    axis.axvline(1.0, color="#666666", linestyle="--", linewidth=0.8)
    axis.axhline(1.0, color="#666666", linestyle="--", linewidth=0.8)
    axis.set_xlabel("R / G")
    axis.set_ylabel("B / G")
    axis.set_title(f"{sample_id} | camera-seed RAW statistics (0..255)")
    legend_labels = ["neutral", "green", "cyan/blue", "blue", "magenta/pink", "warm/yellow"]
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=CLASS_COLORS[label], markersize=8)
        for label in legend_labels
        if label in {info["class"] for info in stats.values()}
    ]
    if handles:
        axis.legend(handles=handles, loc="upper left", frameon=True, fontsize=8)
    axis.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def render_seed59_compare(
    path,
    sample_id,
    seed59_stats,
    legacy_preview,
    source_wb_preview,
    raw_preview,
    viewer_preview,
    legacy_gain,
    self_gain,
    source_wb_gain,
):
    images = [legacy_preview, source_wb_preview, raw_preview, viewer_preview]
    titles = [
        "Legacy preview\nmean-camera RGB gain",
        "Source-wb replay\nreapply Step0 wb_gain",
        "RAW-style preview\ncommon scalar, no RGB gain",
        "Gray-world viewer\nper-image RGB gain",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.8, 3.9), dpi=220)
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(title, fontsize=9, pad=6)
    footer = (
        f"camera seed 59 RAW stats: R/G={seed59_stats['r_over_g']:.3f}, "
        f"B/G={seed59_stats['b_over_g']:.3f} | "
        f"legacy gain={np.round(legacy_gain, 3).tolist()} | "
        f"source wb_gain={np.round(source_wb_gain, 3).tolist()} | "
        f"gray-world gain={np.round(self_gain, 3).tolist()}"
    )
    fig.suptitle(f"{sample_id} | why seed 59 looked pink before", fontsize=14, y=0.995)
    fig.text(0.5, 0.02, footer, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    raw_path = args.source_root / f"{args.sample_id}.npy"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing source npy: {raw_path}")

    raw = np.load(raw_path).astype(np.float32) / 4095.0
    simulator = RAWSimulator()
    wb_gain = compute_wb_gain(raw)

    seed_stats = {}
    raw_arrays = {}
    raw_percentiles = []
    viewer_previews = {}
    viewer_gains = {}
    viewer_scalars = {}
    for seed in range(args.seed_count):
        arr, stats = simulate_camera_seed(simulator, raw, wb_gain, seed, args.bit_depth)
        stats["class"] = classify_seed(stats["r_over_g"], stats["b_over_g"])
        seed_stats[seed] = stats
        raw_arrays[seed] = arr
        raw_percentiles.append(float(np.percentile(np.clip(arr, 0.0, None), 99.5)))
        viewer_preview, viewer_gain, viewer_scalar = make_gray_world_preview(arr)
        viewer_previews[seed] = viewer_preview
        viewer_gains[seed] = [float(v) for v in viewer_gain]
        viewer_scalars[seed] = float(viewer_scalar)

    common_raw_scalar = float(np.median(raw_percentiles))

    raw_previews = {
        seed: make_raw_preview(arr, common_raw_scalar) for seed, arr in raw_arrays.items()
    }

    _mean_camera_arr, mean_camera_rgb = simulate_mean_camera(
        simulator, raw, wb_gain, args.bit_depth
    )
    legacy_gain = np.mean(mean_camera_rgb) / (np.array(mean_camera_rgb) + 1e-6)
    seed59_mean = np.array(seed_stats[59]["mean_rgb"], dtype=np.float32)
    self_gain = seed59_mean.mean() / (seed59_mean + 1e-6)
    legacy_preview, legacy_scalar = make_gain_preview(raw_arrays[59], legacy_gain)
    source_wb_preview, source_wb_scalar = make_gain_preview(raw_arrays[59], wb_gain)

    first_n_seeds = list(range(min(args.first_n, args.seed_count)))
    representative_seeds = [
        seed for seed in args.representative_seeds if 0 <= seed < args.seed_count
    ]

    scatter_path = args.output_root / f"{args.sample_id}_camera_seed_scatter_rawstats_v2.png"
    rep_path = (
        args.output_root / f"{args.sample_id}_camera_seed_examples_raw_viewer_v2.png"
    )
    raw_grid_path = (
        args.output_root / f"{args.sample_id}_camera_seed_first24_rawstyle_v2.png"
    )
    viewer_grid_path = (
        args.output_root / f"{args.sample_id}_camera_seed_first24_viewer_v2.png"
    )
    compare_path = (
        args.output_root / f"{args.sample_id}_camera_seed59_display_compare_v2.png"
    )

    render_scatter(scatter_path, args.sample_id, seed_stats, representative_seeds)
    render_representative_grid(
        rep_path, args.sample_id, representative_seeds, raw_previews, viewer_previews, seed_stats
    )
    render_first_n_grid(
        raw_grid_path,
        args.sample_id,
        "RAW-style preview",
        first_n_seeds,
        raw_previews,
        seed_stats,
    )
    render_first_n_grid(
        viewer_grid_path,
        args.sample_id,
        "Viewer preview",
        first_n_seeds,
        viewer_previews,
        seed_stats,
    )
    render_seed59_compare(
        compare_path,
        args.sample_id,
        seed_stats[59],
        legacy_preview,
        source_wb_preview,
        raw_previews[59],
        viewer_previews[59],
        legacy_gain,
        self_gain,
        wb_gain,
    )

    preview_dir = args.output_root / "single_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for seed in representative_seeds:
        save_rgb(preview_dir / f"{args.sample_id}_camera_seed{seed}_rawstyle_v2.png", raw_previews[seed])
        save_rgb(preview_dir / f"{args.sample_id}_camera_seed{seed}_viewer_v2.png", viewer_previews[seed])

    summary = {
        "sample_id": args.sample_id,
        "source_path": str(raw_path),
        "bit_depth": int(args.bit_depth),
        "camera_seed_definition": "seed only controls RAWSimulator._sample_camera()",
        "raw_style_preview": {
            "formula": "preview = clip(arr / common_raw_scalar, 0, 1) ** (1/2.2)",
            "common_raw_scalar": common_raw_scalar,
        },
        "viewer_preview": {
            "formula": (
                "display_gain = mean(arr.mean(axis=(0,1))) / arr.mean(axis=(0,1)); "
                "preview = clip(arr * display_gain / p99.5(arr * display_gain), 0, 1) ** (1/2.2)"
            ),
            "example_seed59_gain": [float(v) for v in self_gain],
            "example_seed59_scalar": viewer_scalars[59],
        },
        "legacy_preview_for_seed59": {
            "formula": (
                "display_gain = mean(mean_camera_rgb) / mean_camera_rgb; "
                "preview = clip(arr * display_gain / p99.5(arr * display_gain), 0, 1) ** (1/2.2)"
            ),
            "mean_camera_rgb": mean_camera_rgb,
            "legacy_gain": [float(v) for v in legacy_gain],
            "legacy_scalar": legacy_scalar,
        },
        "source_wb_replay_for_seed59": {
            "formula": "preview = clip(arr * source_wb_gain / p99.5(arr * source_wb_gain), 0, 1) ** (1/2.2)",
            "source_wb_gain": [float(v) for v in wb_gain],
            "source_wb_scalar": source_wb_scalar,
        },
        "representative_seeds": representative_seeds,
        "first_n_seeds": first_n_seeds,
        "seed_stats": seed_stats,
        "viewer_gains": viewer_gains,
        "outputs": {
            "scatter": str(scatter_path),
            "representative_grid": str(rep_path),
            "first24_rawstyle": str(raw_grid_path),
            "first24_viewer": str(viewer_grid_path),
            "seed59_compare": str(compare_path),
        },
    }
    summary_path = args.output_root / f"{args.sample_id}_camera_seed_views_v2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["outputs"], indent=2))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
