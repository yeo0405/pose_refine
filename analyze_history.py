#!/usr/bin/env python3
"""Analyze keypoint-refinement training history.json and plot the curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_history(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)
    if not isinstance(history, list) or not history:
        raise ValueError(f"Empty or invalid history file: {path}")
    return history


def series(history, split, key):
    out = []
    for rec in history:
        v = rec.get(split, {}).get(key)
        out.append(float(v) if v is not None else np.nan)
    return np.asarray(out, dtype=np.float64)


def improvement_ratio(history, split):
    """Use the logged ratio; fall back to 1 - pred/init for older logs."""
    ratio = series(history, split, "improvement_ratio")
    if np.isfinite(ratio).any():
        return ratio
    pred = series(history, split, "mean_pixel_error")
    init = series(history, split, "initial_pixel_error")
    return 1.0 - pred / np.where(init > 0, init, np.nan)


def smooth(y, window):
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    padded = np.concatenate(
        [np.full(window // 2, y[0]), y, np.full(window - 1 - window // 2, y[-1])]
    )
    return np.convolve(padded, kernel, mode="valid")


def trend_slope(epochs, y):
    m = np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(np.polyfit(epochs[m], y[m], 1)[0])


def analyze(history, window):
    epochs = np.asarray(
        [r.get("epoch", i + 1) for i, r in enumerate(history)], dtype=np.float64
    )
    n = len(epochs)

    tr_ratio = improvement_ratio(history, "train")
    va_ratio = improvement_ratio(history, "validation")
    tr_pred = series(history, "train", "mean_pixel_error")
    va_pred = series(history, "validation", "mean_pixel_error")
    va_loss = series(history, "validation", "loss")
    tr_loss = series(history, "train", "loss")
    va_worse = series(history, "validation", "worse_ratio")
    tr_worse = series(history, "train", "worse_ratio")

    lines = []
    add = lines.append

    add("=" * 70)
    add("TRAINING HISTORY ANALYSIS")
    add("=" * 70)
    add(f"Epochs logged: {n} (epoch {int(epochs[0])} .. {int(epochs[-1])})")

    # ---- best epoch
    best_i = int(np.nanargmax(va_ratio))
    add("")
    add("-" * 70)
    add("BEST EPOCH (by validation improvement_ratio)")
    add("-" * 70)
    add(
        f"Epoch {int(epochs[best_i])}: val improve {va_ratio[best_i] * 100:+.2f}%, "
        f"val pred error {va_pred[best_i]:.3f}px, val worse {va_worse[best_i] * 100:.1f}%"
    )
    add(
        f"Last epoch ({int(epochs[-1])}): val improve {va_ratio[-1] * 100:+.2f}%, "
        f"val pred error {va_pred[-1]:.3f}px, val worse {va_worse[-1] * 100:.1f}%"
    )
    add(
        f"Best is {n - 1 - best_i} epochs before the end "
        f"({'last.pth ~ best.pth' if n - 1 - best_i < max(5, n // 20) else 'best.pth differs from last.pth'})"
    )

    # ---- convergence
    sm = smooth(va_ratio, window)
    best_sm = np.nanmax(sm)
    plateau_i = int(np.argmax(sm >= best_sm - 0.005))
    add("")
    add("-" * 70)
    add(f"CONVERGENCE (moving average window={window})")
    add("-" * 70)
    add(f"Smoothed val improvement peaks at {best_sm * 100:+.2f}%")
    add(
        f"Reached within 0.5% of the peak at epoch {int(epochs[plateau_i])} "
        f"({plateau_i + 1}/{n} = {(plateau_i + 1) / n * 100:.0f}% of training)"
    )
    if plateau_i + 1 < 0.6 * n:
        add(
            f"-> Later epochs add little; ~{int(epochs[plateau_i]) * 2} epochs would likely be enough."
        )
    else:
        add(
            "-> Improvement continued late in training; more epochs / a longer schedule may still help."
        )

    tail = max(10, n // 5)
    tail_slope = trend_slope(epochs[-tail:], va_ratio[-tail:])
    add(
        f"Val improvement trend over last {tail} epochs: {tail_slope * 100 * 10:+.3f}% per 10 epochs"
    )
    add(
        f"Val improvement noise (std, last {tail}): {np.nanstd(va_ratio[-tail:]) * 100:.3f}%"
    )

    # ---- generalization gap
    gap_px = va_pred - tr_pred
    gap_ratio = tr_ratio - va_ratio
    q = max(1, n // 4)
    add("")
    add("-" * 70)
    add("GENERALIZATION GAP (validation - train)")
    add("-" * 70)
    add(
        f"Pixel error gap : first quarter {np.nanmean(gap_px[:q]):+.3f}px -> "
        f"last quarter {np.nanmean(gap_px[-q:]):+.3f}px"
    )
    add(
        f"Improve-ratio gap (train - val): first quarter {np.nanmean(gap_ratio[:q]) * 100:+.2f}% -> "
        f"last quarter {np.nanmean(gap_ratio[-q:]) * 100:+.2f}%"
    )
    add(
        f"Final: train {tr_pred[-1]:.3f}px vs val {va_pred[-1]:.3f}px "
        f"(val is {va_pred[-1] / max(tr_pred[-1], 1e-9):.2f}x train)"
    )

    final_gap = gap_ratio[-1]
    if final_gap > 0.10:
        add(
            "-> Large gap: clear overfitting (more offset diversity, geometric aug, regularization)."
        )
    elif final_gap > 0.04:
        add(
            "-> Moderate gap: mild overfitting; val is still usable. Scene diversity is the likely limit."
        )
    else:
        add("-> Small gap: little overfitting.")

    # val loss rising while train falls?
    vl_best_i = int(np.nanargmin(smooth(va_loss, window)))
    add(
        f"Val loss minimum (smoothed) at epoch {int(epochs[vl_best_i])}; "
        f"final val loss {va_loss[-1]:.5f} vs min {np.nanmin(smooth(va_loss, window)):.5f}"
    )
    if vl_best_i + 1 < 0.6 * n and va_loss[-1] > 1.1 * np.nanmin(
        smooth(va_loss, window)
    ):
        add(
            "-> Val loss rose after its minimum while train kept falling: overfitting onset."
        )

    # ---- worse ratio
    add("")
    add("-" * 70)
    add("KEYPOINTS MADE WORSE (pred error > init error)")
    add("-" * 70)
    add(
        f"Train worse ratio: {tr_worse[-1] * 100:.1f}%   Val worse ratio: {va_worse[-1] * 100:.1f}%"
    )
    add(
        f"Val worse ratio min: {np.nanmin(va_worse) * 100:.1f}% at epoch {int(epochs[int(np.nanargmin(va_worse))])}"
    )

    # ---- init error consistency
    init_v = series(history, "validation", "initial_pixel_error")
    if np.isfinite(init_v).any():
        add("")
        add("-" * 70)
        add("VALIDATION INITIAL ERROR (should be constant with a fixed val_seed)")
        add("-" * 70)
        add(f"mean {np.nanmean(init_v):.3f}px, std {np.nanstd(init_v):.4f}px")
        if np.nanstd(init_v) > 0.05:
            add(
                "-> Varies across epochs: validation offsets are NOT fixed; improvement_ratio is not strictly comparable."
            )
        else:
            add("-> Constant: validation metric is comparable across epochs.")

    # ---- LR
    lr = (
        series(history, "", "learning_rate")
        if False
        else np.asarray(
            [r.get("learning_rate", np.nan) for r in history], dtype=np.float64
        )
    )

    add("")
    add("-" * 70)
    add("CHECKPOINT ADVICE")
    add("-" * 70)
    add(
        f"Use best.pth (epoch {int(epochs[best_i])}). Differences between top epochs are within noise "
        f"({np.nanstd(va_ratio[-tail:]) * 100:.2f}%)."
    )
    add("=" * 70)

    data = dict(
        epochs=epochs,
        tr_ratio=tr_ratio,
        va_ratio=va_ratio,
        tr_pred=tr_pred,
        va_pred=va_pred,
        tr_loss=tr_loss,
        va_loss=va_loss,
        tr_worse=tr_worse,
        va_worse=va_worse,
        lr=lr,
        tr_init=series(history, "train", "initial_pixel_error"),
        va_init=init_v,
        best_i=best_i,
        gap_px=gap_px,
        gap_ratio=gap_ratio,
        tr_heat=series(history, "train", "heatmap_loss"),
        va_heat=series(history, "validation", "heatmap_loss"),
        tr_coord=series(history, "train", "coordinate_loss"),
        va_coord=series(history, "validation", "coordinate_loss"),
    )
    return "\n".join(lines), data


def plot(data, window, out_path):
    e = data["epochs"]
    best_i = data["best_i"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Keypoint Refinement Training Analysis", fontsize=15)

    def line(ax, y, label, color, raw_alpha=0.25, ls="-"):
        if window > 1:
            ax.plot(e, y, color=color, alpha=raw_alpha, lw=1)
            ax.plot(e, smooth(y, window), color=color, lw=2, ls=ls, label=label)
        else:
            ax.plot(e, y, color=color, lw=2, ls=ls, label=label)

    # 1. Loss
    ax = axes[0, 0]
    line(ax, data["tr_loss"], "train total", "tab:blue")
    line(ax, data["va_loss"], "val total", "tab:red")
    line(ax, data["tr_coord"], "train coord", "tab:cyan", ls="--")
    line(ax, data["va_coord"], "val coord", "tab:orange", ls="--")
    ax.set_yscale("log")
    ax.set_title("Loss (log scale)")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 2. Improvement ratio
    ax = axes[0, 1]
    line(ax, data["tr_ratio"] * 100, "train", "tab:blue")
    line(ax, data["va_ratio"] * 100, "val", "tab:red")
    ax.axvline(e[best_i], color="green", ls=":", label=f"best epoch {int(e[best_i])}")
    ax.scatter([e[best_i]], [data["va_ratio"][best_i] * 100], color="green", zorder=5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Improvement ratio (%)  higher = better")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3. Pixel error
    ax = axes[0, 2]
    line(ax, data["tr_pred"], "train pred", "tab:blue")
    line(ax, data["va_pred"], "val pred", "tab:red")
    if np.isfinite(data["va_init"]).any():
        ax.plot(
            e,
            data["va_init"],
            color="tab:red",
            ls=":",
            lw=1.5,
            label="val init (offset)",
        )
    if np.isfinite(data["tr_init"]).any():
        ax.plot(
            e,
            data["tr_init"],
            color="tab:blue",
            ls=":",
            lw=1.5,
            label="train init (offset)",
        )
    ax.set_title("Mean pixel error (px)")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 4. Worse ratio
    ax = axes[1, 0]
    line(ax, data["tr_worse"] * 100, "train", "tab:blue")
    line(ax, data["va_worse"] * 100, "val", "tab:red")
    ax.set_title("Keypoints made worse (%)")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 5. Generalization gap
    ax = axes[1, 1]
    line(ax, data["gap_px"], "pixel error gap (val - train) [px]", "tab:purple")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("px", color="tab:purple")
    ax2 = ax.twinx()
    ax2.plot(
        e,
        smooth(data["gap_ratio"], window) * 100,
        color="tab:green",
        lw=2,
        label="improve gap (train - val) [%]",
    )
    ax2.set_ylabel("%", color="tab:green")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)
    ax.set_title("Generalization gap")
    ax.grid(alpha=0.3)

    # 6. LR + heatmap loss
    ax = axes[1, 2]
    ax.plot(e, data["lr"], color="tab:gray", lw=2, label="learning rate")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax2 = ax.twinx()
    if np.isfinite(data["tr_heat"]).any():
        ax2.plot(
            e,
            smooth(data["tr_heat"], window),
            color="tab:blue",
            ls="--",
            label="train heatmap loss",
        )
        ax2.plot(
            e,
            smooth(data["va_heat"], window),
            color="tab:red",
            ls="--",
            label="val heatmap loss",
        )
        ax2.set_yscale("log")
        ax2.set_ylabel("heatmap loss")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)
    ax.set_title("Learning rate / heatmap loss")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history", type=str, default="runs/keypoint_refinement/history.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: next to history.json)",
    )
    parser.add_argument(
        "--smooth", type=int, default=5, help="Moving-average window (1 = off)"
    )
    args = parser.parse_args()

    history_path = Path(args.history)
    history = load_history(history_path)

    out_dir = Path(args.output) if args.output else history_path.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, data = analyze(history, max(1, args.smooth))
    print(summary)

    (out_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")

    png = out_dir / "training_analysis.png"
    plot(data, max(1, args.smooth), png)

    print(f"\nSaved: {png}")
    print(f"Saved: {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
