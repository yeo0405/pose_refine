from __future__ import annotations

import argparse
import copy
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from models.keypoint_refinement import build_model_from_config
from train import (
    COND_DIM,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_offset_config,
    build_samples,
    crop_square,
    generate_offset_keypoints,
    get_roi,
    make_gaussian_heatmaps,
    read_config,
    rigid_project,
    split_samples,
)

COLORS = {
    "offset": (0, 165, 255),
    "prediction": (255, 0, 255),
    "gt": (0, 255, 0),
}

# bucket edges (px) for initial offset error
BUCKETS = [(0, 10), (10, 20), (20, 40), (40, float("inf"))]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_offset_pixels(sample, offset_cfg, rng, width, height):
    """Online rigid offset (same code path as training). Returns (4, 2) px."""
    result = None

    if offset_cfg["enabled"]:
        result = generate_offset_keypoints(sample, offset_cfg, rng)

    if result is None:
        return (
            sample["gt_keypoints"][:, :2].astype(np.float64)
            * np.array([width, height]),
            0.0,
        )

    return result[0], float(result[1])


def prepare_input(image_bgr, offset_px, image_size, roi_padding, min_roi_size):
    """Mirror of KeypointDataset.__getitem__ (eval mode, no augmentation)."""
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    center_x, center_y, roi_side = get_roi(
        offset_px, np.ones(4, dtype=bool), roi_padding, min_roi_size
    )

    crop, x0, y0, side = crop_square(image, center_x, center_y, roi_side, image_size)

    offset = np.zeros((4, 3), dtype=np.float32)
    offset[:, 0] = (offset_px[:, 0] - x0) / side
    offset[:, 1] = (offset_px[:, 1] - y0) / side
    offset[:, 2] = 1.0

    crop = crop.astype(np.float32) / 255.0
    crop = (crop - IMAGENET_MEAN) / IMAGENET_STD

    heatmaps = make_gaussian_heatmaps(offset, image_size, sigma=4.0)

    model_input = np.concatenate([crop.transpose(2, 0, 1), heatmaps], axis=0)

    return (
        torch.from_numpy(model_input.astype(np.float32)),
        offset[:, :2].astype(np.float32),
        x0,
        y0,
        side,
    )


def roi_to_pixels(kp_roi, x0, y0, side):
    out = np.zeros((4, 2), dtype=np.float64)
    out[:, 0] = x0 + kp_roi[:, 0] * side
    out[:, 1] = y0 + kp_roi[:, 1] * side
    return out


def draw_keypoints(image, keypoints, color, name, visibility=None):
    for i, point in enumerate(keypoints):
        if visibility is not None and visibility[i] <= 0:
            continue

        x, y = np.round(point[:2]).astype(int)

        cv2.circle(image, (int(x), int(y)), 7, color, 2, cv2.LINE_AA)
        cv2.putText(
            image,
            f"{name}{i}",
            (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_legend(image):
    y = 30

    for name, color in (
        ("Offset", COLORS["offset"]),
        ("Prediction", COLORS["prediction"]),
        ("GT", COLORS["gt"]),
    ):
        cv2.line(image, (20, y), (55, y), color, 3)
        cv2.putText(image, name, (65, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        y += 30


def safe_mean(values):
    return float(np.mean(values)) if len(values) else float("nan")


def summarize(init_err, pred_err):
    """Relative metrics; improvement can be negative."""
    init_sum = float(np.sum(init_err))
    improve = init_err - pred_err

    return {
        "count": len(init_err),
        "init": safe_mean(init_err),
        "pred": safe_mean(pred_err),
        "improve_px": safe_mean(improve),
        "improve_ratio": float(np.sum(improve)) / max(init_sum, 1e-9),
        "worse_ratio": safe_mean((pred_err > init_err).astype(np.float64)),
    }


def fmt_summary(s):
    return (
        f"n={s['count']:<6d} init={s['init']:>7.3f}px  pred={s['pred']:>7.3f}px  "
        f"improve={s['improve_px']:>+7.3f}px ({s['improve_ratio'] * 100:>+7.2f}%)  "
        f"worse={s['worse_ratio'] * 100:>5.1f}%"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--checkpoint", type=str, default="runs/keypoint_refinement/best.pth"
    )
    parser.add_argument(
        "--samples", type=int, default=0, help="0 = evaluate the whole split."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--split", choices=["validation", "train", "all"], default="validation"
    )
    parser.add_argument("--output", type=str, default="runs/keypoint_refinement/test")
    parser.add_argument(
        "--use-config-offset",
        action="store_true",
        help="Use the offset ranges in --config instead of those stored in the checkpoint.",
    )
    parser.add_argument(
        "--full-offset",
        action="store_true",
        help="Disable random_scale (always full-magnitude offset).",
    )
    parser.add_argument(
        "--offset-scale",
        type=float,
        default=1.0,
        help="Global multiplier on the whole offset range (e.g. 0.5 = small offsets).",
    )
    parser.add_argument(
        "--val-seed",
        type=int,
        default=None,
        help="Override offset seed base (default: val_seed from offset config).",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=None,
        help="Override the number of cascade stages (new architecture only).",
    )
    parser.add_argument(
        "--max-debug-images",
        type=int,
        default=200,
        help="Only save this many debug images (0 = none, -1 = all).",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    config = read_config(Path(args.config))

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    ckpt_cfg = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    ckpt_cfg = ckpt_cfg or {}

    # ---- model / preprocessing settings come from the checkpoint when possible
    # (a legacy checkpoint has no "arch" and must NOT inherit it from config.yaml)
    if ckpt_cfg.get("model"):
        model_cfg = copy.deepcopy(ckpt_cfg["model"])
    else:
        model_cfg = copy.deepcopy(config["model"])

    dataset_cfg = copy.deepcopy(config["dataset"])

    for key in ("image_size", "roi_padding", "min_roi_size"):
        if key in (ckpt_cfg.get("dataset") or {}):
            dataset_cfg[key] = ckpt_cfg["dataset"][key]

    image_size = int(dataset_cfg["image_size"])
    roi_padding = float(dataset_cfg["roi_padding"])
    min_roi_size = int(dataset_cfg["min_roi_size"])

    if ckpt_cfg.get("offset") and not args.use_config_offset:
        offset_cfg = build_offset_config({"offset": ckpt_cfg["offset"]})
    else:
        offset_cfg = build_offset_config(config)

    if args.full_offset:
        offset_cfg["random_scale"] = False

    if args.val_seed is not None:
        offset_cfg["val_seed"] = int(args.val_seed)

    cond_dim = int(
        model_cfg.get(
            "cond_dim", COND_DIM if model_cfg.get("use_size_condition", True) else 0
        )
    )

    # samples need cond built with the same size_scale_mm as training
    config_for_samples = copy.deepcopy(config)
    config_for_samples["model"]["size_scale_mm"] = float(
        model_cfg.get("size_scale_mm", config["model"].get("size_scale_mm", 300.0))
    )

    output_dir = Path(args.output)
    debug_dir = output_dir / "debug_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_debug_images != 0:
        debug_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    model = build_model_from_config(model_cfg, cond_dim=cond_dim, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    is_cascade = hasattr(model, "num_iters")
    arch = model_cfg.get("arch") or "legacy (pooled regression)"
    num_iters = (
        int(args.num_iters)
        if (args.num_iters is not None and is_cascade)
        else (int(model.num_iters) if is_cascade else 1)
    )

    print(f"[INFO] Device     : {device}")
    print(f"[INFO] Checkpoint : {args.checkpoint}")
    print(f"[INFO] Epoch      : {checkpoint.get('epoch', 'unknown')}")
    print(f"[INFO] Arch       : {arch}")
    print(f"[INFO] cond_dim   : {cond_dim}")
    print(f"[INFO] Cascade it.: {num_iters}")
    print(
        f"[INFO] ROI        : image_size={image_size} roi_padding={roi_padding} "
        f"min_roi_size={min_roi_size}"
    )
    print(
        "[INFO] Offset cfg : "
        f"t={offset_cfg['translation_mm']} r={offset_cfg['rotation_deg']} "
        f"random_scale={offset_cfg['random_scale']} min_scale={offset_cfg['min_scale']} "
        f"val_seed={offset_cfg['val_seed']} global_scale={args.offset_scale}"
    )

    samples = build_samples(config_for_samples)

    train_samples, val_samples = split_samples(
        samples,
        float(dataset_cfg["train_ratio"]),
        int(dataset_cfg["split_seed"]),
    )

    print(f"[INFO] Train samples      : {len(train_samples)}")
    print(f"[INFO] Validation samples : {len(val_samples)}")

    if args.split == "validation":
        pool = val_samples
    elif args.split == "train":
        pool = train_samples
    else:
        pool = samples

    # keep the pool index: the offset seed is val_seed + index (same as training validation)
    indexed = list(enumerate(pool))
    random.Random(args.seed).shuffle(indexed)

    selected = indexed[: args.samples] if args.samples > 0 else indexed

    print(f"[INFO] Selected split : {args.split}")
    print(f"[INFO] Testing samples: {len(selected)}")

    if not selected:
        raise RuntimeError("No samples selected.")

    csv_path = output_dir / "predictions.csv"

    rows = []

    # flat per-keypoint records (valid keypoints only)
    rec_init, rec_pred, rec_stage1, rec_kp = [], [], [], []
    rec_gt_delta, rec_pred_delta, rec_scale = [], [], []

    # rigid-body consistency (per instance with 4 valid keypoints)
    rig_resid, rig_proj_err, rig_raw_err = [], [], []

    saved_debug = 0

    with torch.no_grad():
        for order, (pool_index, sample) in enumerate(selected):
            image = cv2.imread(str(sample["image_path"]))

            if image is None:
                print(f"[WARN] Cannot read image: {sample['image_path']}")
                continue

            height, width = image.shape[:2]

            rng = np.random.default_rng(int(offset_cfg["val_seed"]) + pool_index)

            if args.offset_scale != 1.0:
                result = generate_offset_keypoints(
                    sample, offset_cfg, rng, global_scale=args.offset_scale
                )

                if result is None:
                    offset_pixels = sample["gt_keypoints"][:, :2].astype(
                        np.float64
                    ) * np.array([width, height])
                    offset_scale = 0.0
                else:
                    offset_pixels, offset_scale = result[0], float(result[1])
            else:
                offset_pixels, offset_scale = make_offset_pixels(
                    sample, offset_cfg, rng, width, height
                )

            model_input, initial_coords, x0, y0, side = prepare_input(
                image, offset_pixels, image_size, roi_padding, min_roi_size
            )

            model_input = model_input.unsqueeze(0).to(device)
            initial_t = torch.from_numpy(initial_coords).float().unsqueeze(0).to(device)

            cond = None

            if cond_dim > 0:
                cond = torch.from_numpy(sample["cond"]).float().unsqueeze(0).to(device)

            if is_cascade:
                output = model(model_input, initial_t, cond, num_iters=num_iters)
            else:
                output = model(model_input, initial_t, cond)

            pred_roi = output["keypoints"][0].float().cpu().numpy()
            pred_pixels = roi_to_pixels(pred_roi, x0, y0, side)

            stages = output.get("stages")

            if stages:
                stage1_pixels = roi_to_pixels(
                    stages[0]["keypoints"][0].float().cpu().numpy(), x0, y0, side
                )
            else:
                stage1_pixels = pred_pixels

            gt_kps = sample["gt_keypoints"]
            gt_pixels = gt_kps[:, :2].astype(np.float64).copy()
            gt_pixels[:, 0] *= width
            gt_pixels[:, 1] *= height

            valid = gt_kps[:, 2] > 0

            gt_delta = gt_pixels - offset_pixels
            pred_delta = pred_pixels - offset_pixels

            init_err = np.linalg.norm(offset_pixels - gt_pixels, axis=1)
            pred_err = np.linalg.norm(pred_pixels - gt_pixels, axis=1)
            stage1_err = np.linalg.norm(stage1_pixels - gt_pixels, axis=1)
            improve = init_err - pred_err

            if valid.any():
                inst_init = float(init_err[valid].mean())
                inst_pred = float(pred_err[valid].mean())
                inst_improve = float(improve[valid].mean())
            else:
                inst_init = inst_pred = inst_improve = float("nan")

            # ---- rigid-body consistency of the final prediction
            rigid_resid_px = rigid_proj_px = float("nan")

            if valid.all():
                proj = rigid_project(pred_pixels, sample["K"], sample["kp3d"])

                if proj is not None:
                    rigid_resid_px = float(
                        np.linalg.norm(proj - pred_pixels, axis=1).mean()
                    )
                    rigid_proj_px = float(
                        np.linalg.norm(proj - gt_pixels, axis=1).mean()
                    )

                    rig_resid.append(rigid_resid_px)
                    rig_proj_err.append(rigid_proj_px)
                    rig_raw_err.append(inst_pred)

            row = {
                "image": str(sample["relative_path"]),
                "instance_index": sample["instance_index"],
                "obj_id": sample["obj_id"],
                "pool_index": pool_index,
                "offset_scale": offset_scale,
                "valid_keypoints": int(valid.sum()),
                "init_error_px": inst_init,
                "pred_error_px": inst_pred,
                "stage1_error_px": (
                    float(stage1_err[valid].mean()) if valid.any() else float("nan")
                ),
                "improvement_px": inst_improve,
                "rigid_residual_px": rigid_resid_px,
                "rigid_projected_error_px": rigid_proj_px,
            }

            for kp in range(4):
                ok = bool(valid[kp])

                row[f"kp{kp}_valid"] = int(ok)
                row[f"kp{kp}_init_error_px"] = float(init_err[kp]) if ok else ""
                row[f"kp{kp}_stage1_error_px"] = float(stage1_err[kp]) if ok else ""
                row[f"kp{kp}_pred_error_px"] = float(pred_err[kp]) if ok else ""
                row[f"kp{kp}_improvement_px"] = float(improve[kp]) if ok else ""
                row[f"kp{kp}_gt_dx"] = float(gt_delta[kp, 0]) if ok else ""
                row[f"kp{kp}_gt_dy"] = float(gt_delta[kp, 1]) if ok else ""
                row[f"kp{kp}_pred_dx"] = float(pred_delta[kp, 0]) if ok else ""
                row[f"kp{kp}_pred_dy"] = float(pred_delta[kp, 1]) if ok else ""

                if ok:
                    rec_init.append(init_err[kp])
                    rec_pred.append(pred_err[kp])
                    rec_stage1.append(stage1_err[kp])
                    rec_kp.append(kp)
                    rec_gt_delta.append(gt_delta[kp])
                    rec_pred_delta.append(pred_delta[kp])
                    rec_scale.append(offset_scale)

            rows.append(row)

            want_debug = (
                args.max_debug_images < 0 or saved_debug < args.max_debug_images
            )

            if want_debug:
                debug = image.copy()

                for kp in np.where(valid)[0]:
                    start = tuple(int(v) for v in np.round(offset_pixels[kp]))
                    pred_end = tuple(int(v) for v in np.round(pred_pixels[kp]))
                    gt_end = tuple(int(v) for v in np.round(gt_pixels[kp]))

                    cv2.arrowedLine(
                        debug,
                        start,
                        pred_end,
                        COLORS["prediction"],
                        2,
                        cv2.LINE_AA,
                        tipLength=0.2,
                    )
                    cv2.line(debug, start, gt_end, COLORS["gt"], 1, cv2.LINE_AA)

                draw_keypoints(debug, offset_pixels, COLORS["offset"], "O")
                draw_keypoints(debug, pred_pixels, COLORS["prediction"], "P")
                draw_keypoints(
                    debug, gt_pixels, COLORS["gt"], "G", visibility=gt_kps[:, 2]
                )
                draw_legend(debug)

                cv2.putText(
                    debug,
                    f"Init: {inst_init:.2f}px  Pred: {inst_pred:.2f}px  "
                    f"Improve: {inst_improve:+.2f}px",
                    (20, height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    COLORS["prediction"] if inst_improve >= 0 else (0, 0, 255),
                    2,
                )

                name = (
                    f"{order:04d}_{sample['image_path'].stem}_"
                    f"{sample['instance_index']}.jpg"
                )
                cv2.imwrite(str(debug_dir / name), debug)
                saved_debug += 1

            print(
                f"[{order + 1:04d}/{len(selected):04d}] {sample['relative_path']} | "
                f"init={inst_init:.2f}px pred={inst_pred:.2f}px "
                f"improve={inst_improve:+.2f}px"
            )

    if not rows:
        raise RuntimeError("No valid predictions were generated.")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    init_arr = np.asarray(rec_init, dtype=np.float64)
    pred_arr = np.asarray(rec_pred, dtype=np.float64)
    stage1_arr = np.asarray(rec_stage1, dtype=np.float64)
    kp_arr = np.asarray(rec_kp, dtype=np.int64)
    gt_d = np.asarray(rec_gt_delta, dtype=np.float64)
    pr_d = np.asarray(rec_pred_delta, dtype=np.float64)
    scale_arr = np.asarray(rec_scale, dtype=np.float64)

    print("\n" + "=" * 78)
    print("TEST SUMMARY")
    print("=" * 78)
    print(f"Split           : {args.split}")
    print(f"Arch            : {arch}   cascade iterations: {num_iters}")
    print(f"Valid keypoints : {len(init_arr)}")
    print(f"Overall         : {fmt_summary(summarize(init_arr, pred_arr))}")

    if is_cascade and num_iters > 1:
        print(
            f"Per stage       : init {init_arr.mean():.3f} -> "
            f"stage1 {stage1_arr.mean():.3f} -> final {pred_arr.mean():.3f}px"
        )

    print(
        f"PCK             : <3px {np.mean(pred_arr < 3) * 100:.1f}%   "
        f"<5px {np.mean(pred_arr < 5) * 100:.1f}%   "
        f"<10px {np.mean(pred_arr < 10) * 100:.1f}%   "
        f"(median error {np.median(pred_arr):.3f}px, p90 {np.percentile(pred_arr, 90):.3f}px)"
    )

    inst_improve_vals = np.asarray(
        [r["improvement_px"] for r in rows if np.isfinite(r["improvement_px"])]
    )
    if len(inst_improve_vals):
        print(
            f"Per-instance    : mean improvement {inst_improve_vals.mean():+.3f}px, "
            f"median {np.median(inst_improve_vals):+.3f}px, "
            f"worse instances {np.mean(inst_improve_vals < 0) * 100:.1f}%"
        )

    if rig_resid:
        print(
            f"Rigid (IPPE)    : residual {np.mean(rig_resid):.3f}px | "
            f"error raw {np.mean(rig_raw_err):.3f} -> projected "
            f"{np.mean(rig_proj_err):.3f}px (n={len(rig_resid)})"
        )

    print("\n" + "-" * 78)
    print("BY INITIAL ERROR (bucketed by offset magnitude)")
    print("-" * 78)

    for lo, hi in BUCKETS:
        m = (init_arr >= lo) & (init_arr < hi)

        if not m.any():
            continue

        label = f"[{lo:g}, {hi:g})" if np.isfinite(hi) else f"[{lo:g}, inf)"
        print(f"{label:<12} {fmt_summary(summarize(init_arr[m], pred_arr[m]))}")

        if is_cascade and num_iters > 1:
            print(
                f"{'':<12} stage1 {stage1_arr[m].mean():.3f}px -> "
                f"final {pred_arr[m].mean():.3f}px"
            )

    print("\n" + "-" * 78)
    print("PER-KEYPOINT")
    print("-" * 78)

    for kp in range(4):
        m = kp_arr == kp

        if not m.any():
            continue

        print(f"KP{kp:<10} {fmt_summary(summarize(init_arr[m], pred_arr[m]))}")

    print("\n" + "-" * 78)
    print("DELTA ANALYSIS (GT delta = offset -> GT, Pred delta = offset -> prediction)")
    print("-" * 78)

    gt_mag = np.linalg.norm(gt_d, axis=1)
    pr_mag = np.linalg.norm(pr_d, axis=1)

    print(f"GT Delta Mean         : {gt_mag.mean():.3f} px (std {gt_mag.std():.3f})")
    print(f"Pred Delta Mean       : {pr_mag.mean():.3f} px (std {pr_mag.std():.3f})")
    print(f"Delta Magnitude Ratio : {pr_mag.mean() / max(gt_mag.mean(), 1e-8):.3f}")

    valid_dir = gt_mag >= 1.0

    if valid_dir.any():
        dot = np.sum(gt_d[valid_dir] * pr_d[valid_dir], axis=1)
        cosine = dot / np.maximum(gt_mag[valid_dir] * pr_mag[valid_dir], 1e-8)
        projection = dot / np.maximum(gt_mag[valid_dir] ** 2, 1e-8)
        mag_ratio = pr_mag[valid_dir] / gt_mag[valid_dir]

        print(
            f"Direction Accuracy    : {np.mean(cosine > 0) * 100:.2f}% "
            f"({int(np.sum(cosine > 0))}/{len(cosine)})"
        )
        print(f"Mean Direction Cosine : {np.mean(cosine):.4f}")
        print(
            f"Mean Projection Ratio : {np.mean(projection):.4f}  (1.0 = full correction)"
        )
        print(f"Mean Magnitude Ratio  : {np.mean(mag_ratio):.4f}")

    # residual error vector after correction: bias here means a systematic shift
    residual = pr_d - gt_d  # prediction - GT, in px

    print("\n" + "-" * 78)
    print("DELTA COMPONENTS / RESIDUAL BIAS")
    print("-" * 78)
    print(
        f"GT Delta X Mean   : {gt_d[:, 0].mean():+.3f} px   Y: {gt_d[:, 1].mean():+.3f} px"
    )
    print(
        f"Pred Delta X Mean : {pr_d[:, 0].mean():+.3f} px   Y: {pr_d[:, 1].mean():+.3f} px"
    )
    print(
        f"Residual X Mean   : {residual[:, 0].mean():+.3f} px   Y: {residual[:, 1].mean():+.3f} px  (should be ~0)"
    )
    print(
        f"Residual X Std    : {residual[:, 0].std():.3f} px   Y: {residual[:, 1].std():.3f} px"
    )

    if offset_cfg["random_scale"] and np.any(scale_arr > 0):
        print("\n" + "-" * 78)
        print("BY OFFSET SCALE (random_scale factor)")
        print("-" * 78)

        top = float(scale_arr.max()) + 1e-6
        edges = np.linspace(0.0, top, 5)

        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (scale_arr >= lo) & (scale_arr < hi + 1e-9)

            if m.any():
                print(
                    f"[{lo:.2f}, {hi:.2f}) "
                    f"{fmt_summary(summarize(init_arr[m], pred_arr[m]))}"
                )

    print("\n" + "=" * 78)
    print(f"Debug Images  : {debug_dir if args.max_debug_images != 0 else 'disabled'}")
    print(f"Prediction CSV: {csv_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
