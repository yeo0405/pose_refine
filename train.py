from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.keypoint_refinement import build_model_from_config

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

COND_DIM = 6  # 3 normalized sizes + 3 log aspect ratios

OFFSET_DEFAULTS = {
    "enabled": True,
    "translation_mm": {"x": [-20.0, 20.0], "y": [-20.0, 20.0], "z": [-20.0, 20.0]},
    "rotation_deg": {"x": [-10.0, 10.0], "y": [-10.0, 10.0], "z": [-10.0, 10.0]},
    "random_scale": True,
    "min_scale": 0.0,
    "val_seed": 12345,
}

CURRICULUM_DEFAULTS = {
    "enabled": True,
    "start_scale": 1.0,
    "min_scale": 0.15,  # floor of the global offset scale
    "shrink_factor": 0.85,  # scale *= shrink_factor when the trigger fires
    "trigger_improvement": 0.60,  # train improvement_ratio needed to shrink
    "patience": 3,  # consecutive epochs above the trigger
    "keep_full_prob": 0.2,  # prob. of still sampling a full-scale offset
}

EVAL_DEFAULTS = {
    "small_scale": 0.5,  # global scale of the second (small-offset) validation set; 0 = off
    "rigid": True,  # compute rigid-body consistency metrics on validation
}


def merge_section(defaults, user):
    out = copy.deepcopy(defaults)
    for k, v in (user or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def build_offset_config(cfg):
    offset = merge_section(OFFSET_DEFAULTS, cfg.get("offset"))

    for group in ("translation_mm", "rotation_deg"):
        for axis in "xyz":
            lo, hi = offset[group][axis]
            if lo > hi:
                raise ValueError(f"Invalid offset range {group}.{axis}: {[lo, hi]}")

    return offset


def build_condition(size_mm, scale_mm):
    """Object-size condition vector from the PLY bbox size (mm)."""
    sx, sy, sz = [max(float(v), 1e-6) for v in size_mm]

    return np.array(
        [
            sx / scale_mm,
            sy / scale_mm,
            sz / scale_mm,
            math.log(sx / sy),
            math.log(sz / sx),
            math.log(sz / sy),
        ],
        dtype=np.float32,
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_config(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_label_file(path: Path):
    instances = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) < 17:
                raise ValueError(
                    f"Invalid label format: {path}:{line_number}, "
                    f"expected at least 17 values, got {len(values)}"
                )

            values = [float(x) for x in values[:17]]

            instances.append(
                {
                    "obj_id": int(values[0]),
                    "bbox": np.asarray(values[1:5], dtype=np.float32),
                    "keypoints": np.asarray(values[5:17], dtype=np.float32).reshape(
                        4, 3
                    ),
                }
            )

    return instances


def find_image(image_root: Path, relative_label_path: Path):
    image_parent = image_root / relative_label_path.parent

    for ext in IMAGE_EXTENSIONS:
        candidate = image_parent / f"{relative_label_path.stem}{ext}"

        if candidate.is_file():
            return candidate

    return None


def build_samples(cfg):
    root = Path(cfg["dataset"]["root"]).expanduser()

    image_root = root / cfg["dataset"]["image_dir"]
    gt_root = root / cfg["dataset"]["gt_label_dir"]
    meta_root = root / cfg["dataset"]["meta_dir"]

    for name, p in (("Image", image_root), ("GT label", gt_root), ("Meta", meta_root)):
        if not p.is_dir():
            raise FileNotFoundError(f"{name} directory not found: {p}")

    gt_files = sorted(gt_root.rglob("*.txt"))

    if not gt_files:
        raise RuntimeError(f"No GT label files found: {gt_root}")

    size_scale = float(cfg["model"].get("size_scale_mm", 300.0))

    samples = []
    missing_images = 0
    missing_meta = 0

    for gt_path in gt_files:
        relative_path = gt_path.relative_to(gt_root)
        meta_path = (meta_root / relative_path).with_suffix(".json")

        if not meta_path.is_file():
            missing_meta += 1
            continue

        image_path = find_image(image_root, relative_path)

        if image_path is None:
            missing_images += 1
            continue

        gt_instances = read_label_file(gt_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        meta_instances = meta["instances"]

        if len(gt_instances) != len(meta_instances):
            raise ValueError(
                f"Instance count mismatch:\nGT: {gt_path}\nMeta: {meta_path}"
            )

        K = np.asarray(meta["cam_K"], dtype=np.float64).reshape(3, 3)

        for index, (gt, m) in enumerate(zip(gt_instances, meta_instances)):
            if gt["obj_id"] != int(m["obj_id"]):
                raise ValueError(f"Object ID mismatch: {gt_path}, instance {index}")

            size_mm = np.asarray(m["model_size_mm"], dtype=np.float64).reshape(3)

            samples.append(
                {
                    "image_path": image_path,
                    "relative_path": relative_path,
                    "scene": (
                        relative_path.parts[0]
                        if len(relative_path.parts) > 1
                        else "__root__"
                    ),
                    "instance_index": index,
                    "obj_id": gt["obj_id"],
                    "gt_keypoints": gt["keypoints"].copy(),
                    "K": K,
                    "R": np.asarray(m["R_m2c"], dtype=np.float64).reshape(3, 3),
                    "t": np.asarray(m["t_m2c_mm"], dtype=np.float64).reshape(3),
                    "kp3d": np.asarray(m["keypoints_3d_mm"], dtype=np.float64).reshape(
                        4, 3
                    ),
                    "center": np.asarray(
                        m["model_center_mm"], dtype=np.float64
                    ).reshape(3),
                    "size_mm": size_mm,
                    "cond": build_condition(size_mm, size_scale),
                }
            )

    print(f"Label files       : {len(gt_files)}")
    print(f"Training instances: {len(samples)}")
    print(f"Missing meta      : {missing_meta}")
    print(f"Missing images    : {missing_images}")

    if not samples:
        raise RuntimeError("No valid training samples found.")

    sizes = {}
    for s in samples:
        sizes.setdefault(s["obj_id"], s["size_mm"])
    print("Object sizes (mm):")
    for obj_id, size in sorted(sizes.items()):
        print(f"  obj_id={obj_id}: {np.round(size, 2).tolist()}")

    return samples


def split_samples(samples, train_ratio, seed):
    groups = {}

    for sample in samples:
        groups.setdefault(sample["scene"], []).append(sample)

    group_names = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_names)

    if len(group_names) >= 2:
        n = int(round(len(group_names) * train_ratio))
        n = max(1, min(n, len(group_names) - 1))
        train_groups = set(group_names[:n])

        train_samples = [s for s in samples if s["scene"] in train_groups]
        val_samples = [s for s in samples if s["scene"] not in train_groups]
    else:
        image_groups = {}

        for sample in samples:
            image_groups.setdefault(str(sample["relative_path"]), []).append(sample)

        image_names = sorted(image_groups.keys())
        rng.shuffle(image_names)

        if len(image_names) < 2:
            raise RuntimeError(
                "Only one scene and one image were found. "
                "Cannot create an independent validation set."
            )

        n = int(round(len(image_names) * train_ratio))
        n = max(1, min(n, len(image_names) - 1))
        train_images = set(image_names[:n])

        train_samples = [s for s in samples if str(s["relative_path"]) in train_images]
        val_samples = [
            s for s in samples if str(s["relative_path"]) not in train_images
        ]

    if not train_samples or not val_samples:
        raise RuntimeError("Training or validation dataset is empty.")

    return train_samples, val_samples


# ---------------------------------------------------------------------------
# Online rigid-body offset
# ---------------------------------------------------------------------------


def euler_to_rotation(rx, ry, rz):
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)

    return Rz @ Ry @ Rx


def generate_offset_keypoints(sample, offset_cfg, rng, global_scale=1.0, max_tries=10):
    """
    Perturb the GT pose with a random rigid transform and reproject the
    4 face corners. `global_scale` shrinks the whole offset range (curriculum).
    Returns (uv_px (4, 2), scale) or None on failure.
    """
    K, R, t = sample["K"], sample["R"], sample["t"]
    kp3d, center = sample["kp3d"], sample["center"]

    for _ in range(max_tries):
        if offset_cfg["random_scale"]:
            scale = rng.uniform(float(offset_cfg["min_scale"]), 1.0)
        else:
            scale = 1.0

        scale *= float(global_scale)

        t_delta = (
            np.array(
                [rng.uniform(*offset_cfg["translation_mm"][a]) for a in "xyz"],
                dtype=np.float64,
            )
            * scale
        )

        angles = (
            np.deg2rad([rng.uniform(*offset_cfg["rotation_deg"][a]) for a in "xyz"])
            * scale
        )

        R_delta = euler_to_rotation(*angles)

        points = (R_delta @ (kp3d - center).T).T + center + t_delta
        points_cam = (R @ points.T).T + t.reshape(1, 3)

        if not np.all(points_cam[:, 2] > 1e-6):
            continue

        uv = (K @ points_cam.T).T
        uv = uv[:, :2] / uv[:, 2:3]

        return uv, scale

    return None


# ---------------------------------------------------------------------------
# Rigid-body consistency (PnP on the 4 coplanar corners)
# ---------------------------------------------------------------------------


def rigid_project(points_px, K, kp3d):
    """
    Fit a pose to 4 image points with IPPE (planar PnP) and reproject.
    The 4 corners belong to one rigid rectangle, so any residual measures how
    far the points are from being a valid perspective view of it.
    Returns (4, 2) reprojected points or None on failure.
    """
    try:
        ok, rvec, tvec = cv2.solvePnP(
            np.ascontiguousarray(kp3d, dtype=np.float64),
            np.ascontiguousarray(points_px, dtype=np.float64).reshape(-1, 1, 2),
            np.ascontiguousarray(K, dtype=np.float64),
            None,
            flags=cv2.SOLVEPNP_IPPE,
        )

        if not ok:
            return None

        proj, _ = cv2.projectPoints(
            np.ascontiguousarray(kp3d, dtype=np.float64),
            rvec,
            tvec,
            np.ascontiguousarray(K, dtype=np.float64),
            None,
        )

        return proj.reshape(-1, 2)
    except cv2.error:
        return None


# ---------------------------------------------------------------------------
# ROI / crop / augmentation
# ---------------------------------------------------------------------------


def get_roi(points_px, valid, padding, min_size):
    pts = points_px[valid] if valid.sum() >= 2 else points_px

    x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
    y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())

    center_x = (x_min + x_max) * 0.5
    center_y = (y_min + y_max) * 0.5

    roi_side = max(x_max - x_min, y_max - y_min, float(min_size)) * (
        1.0 + 2.0 * float(padding)
    )

    return center_x, center_y, roi_side


def crop_square(image, center_x, center_y, side, output_size):
    side = max(1, int(round(side)))

    x0 = int(math.floor(center_x - side * 0.5))
    y0 = int(math.floor(center_y - side * 0.5))

    pad = side + 2

    padded = cv2.copyMakeBorder(
        image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    crop = padded[y0 + pad : y0 + pad + side, x0 + pad : x0 + pad + side]

    if crop.shape[0] != side or crop.shape[1] != side:
        raise RuntimeError("ROI crop size mismatch.")

    crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

    return crop, float(x0), float(y0), float(side)


def apply_augmentation(image, cfg):
    if not cfg.get("enabled", False):
        return image

    image = image.astype(np.float32)

    brightness = float(cfg.get("brightness", 0.15))
    contrast = float(cfg.get("contrast", 0.15))
    saturation = float(cfg.get("saturation", 0.10))

    if brightness > 0:
        image += random.uniform(-brightness, brightness) * 255.0

    if contrast > 0:
        factor = random.uniform(1.0 - contrast, 1.0 + contrast)
        mean = image.mean(axis=(0, 1), keepdims=True)
        image = (image - mean) * factor + mean

    image = np.clip(image, 0, 255).astype(np.uint8)

    if saturation > 0:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        factor = random.uniform(1.0 - saturation, 1.0 + saturation)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    noise_std = float(cfg.get("noise_std", 0.0))

    if noise_std > 0:
        noise = np.random.normal(0, noise_std * 255.0, image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < float(cfg.get("blur_probability", 0.0)):
        image = cv2.GaussianBlur(image, (3, 3), 0)

    return image


def make_gaussian_heatmaps(keypoints, resolution, sigma):
    heatmaps = np.zeros((4, resolution, resolution), dtype=np.float32)

    for i, (x, y, visibility) in enumerate(keypoints):
        if visibility <= 0:
            continue

        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            continue

        px = x * (resolution - 1)
        py = y * (resolution - 1)

        radius = max(1, int(math.ceil(3.0 * sigma)))

        x0 = max(0, int(math.floor(px)) - radius)
        x1 = min(resolution, int(math.floor(px)) + radius + 1)
        y0 = max(0, int(math.floor(py)) - radius)
        y1 = min(resolution, int(math.floor(py)) + radius + 1)

        yy, xx = np.mgrid[y0:y1, x0:x1]

        heatmaps[i, y0:y1, x0:x1] = np.exp(
            -((xx - px) ** 2 + (yy - py) ** 2) / (2.0 * sigma * sigma)
        )

    return heatmaps


class KeypointDataset(Dataset):
    """
    global_scale : multiplies the whole offset range (curriculum). Updated from
                   the main process before each epoch. DataLoader workers are
                   re-created every epoch (persistent_workers=False), so they
                   always see the current value.
    full_prob    : probability of ignoring global_scale and using scale 1.0
                   (keeps large-offset ability during the curriculum).
    """

    def __init__(self, samples, cfg, training=False, global_scale=1.0, full_prob=0.0):
        self.samples = samples
        self.cfg = cfg
        self.training = training
        self.global_scale = float(global_scale)
        self.full_prob = float(full_prob)

        self.image_size = int(cfg["dataset"]["image_size"])
        self.heatmap_size = int(cfg["dataset"]["heatmap_size"])
        self.padding = float(cfg["dataset"]["roi_padding"])
        self.min_roi_size = int(cfg["dataset"]["min_roi_size"])

        self.augmentation = cfg.get("augmentation", {})
        self.offset_cfg = build_offset_config(cfg)

    def __len__(self):
        return len(self.samples)

    def _make_rng(self, index):
        if self.training:
            return np.random.default_rng(random.getrandbits(64))

        return np.random.default_rng(int(self.offset_cfg["val_seed"]) + index)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = cv2.imread(str(sample["image_path"]), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Cannot read image: {sample['image_path']}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        gt = sample["gt_keypoints"].copy()

        rng = self._make_rng(index)
        result = None

        scale_g = self.global_scale

        if self.training and self.full_prob > 0 and rng.random() < self.full_prob:
            scale_g = 1.0

        if self.offset_cfg["enabled"]:
            result = generate_offset_keypoints(
                sample, self.offset_cfg, rng, global_scale=scale_g
            )

        if result is None:
            offset_px = gt[:, :2].astype(np.float64) * np.array([width, height])
        else:
            offset_px = result[0]

        offset_valid = np.ones(4, dtype=bool)

        center_x, center_y, roi_side = get_roi(
            offset_px, offset_valid, self.padding, self.min_roi_size
        )

        crop, x0, y0, roi_side = crop_square(
            image, center_x, center_y, roi_side, self.image_size
        )

        scale = self.image_size / roi_side

        offset = np.zeros((4, 3), dtype=np.float32)
        offset[:, 0] = (offset_px[:, 0] - x0) * scale / self.image_size
        offset[:, 1] = (offset_px[:, 1] - y0) * scale / self.image_size
        offset[:, 2] = 1.0

        gt[:, 0] = (gt[:, 0] * width - x0) * scale / self.image_size
        gt[:, 1] = (gt[:, 1] * height - y0) * scale / self.image_size

        if self.training:
            crop = apply_augmentation(crop, self.augmentation)

        crop = crop.astype(np.float32) / 255.0
        crop = (crop - IMAGENET_MEAN) / IMAGENET_STD

        input_heatmaps = make_gaussian_heatmaps(offset, self.image_size, sigma=4.0)

        input_tensor = np.concatenate([crop.transpose(2, 0, 1), input_heatmaps], axis=0)

        target_heatmaps = make_gaussian_heatmaps(gt, self.heatmap_size, sigma=1.5)

        initial_coords = offset[:, :2].astype(np.float32)
        target_coords = gt[:, :2].astype(np.float32)

        visibility = (gt[:, 2] > 0).astype(np.float32)
        heatmap_valid = visibility.copy()

        for i in range(4):
            x, y = target_coords[i]

            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                heatmap_valid[i] = 0.0

        return {
            "input": torch.from_numpy(input_tensor.astype(np.float32)),
            "initial_coords": torch.from_numpy(initial_coords),
            "target_coords": torch.from_numpy(target_coords),
            "target_heatmaps": torch.from_numpy(target_heatmaps),
            "visibility": torch.from_numpy(visibility),
            "heatmap_valid": torch.from_numpy(heatmap_valid),
            "roi_side": torch.tensor(roi_side, dtype=torch.float32),
            "roi_x0": torch.tensor(x0, dtype=torch.float32),
            "roi_y0": torch.tensor(y0, dtype=torch.float32),
            "K": torch.from_numpy(sample["K"].astype(np.float32)),
            "kp3d": torch.from_numpy(sample["kp3d"].astype(np.float32)),
            "cond": torch.from_numpy(sample["cond"].copy()),
            "image_path": str(sample["image_path"]),
            "obj_id": torch.tensor(sample["obj_id"], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def coordinate_loss_px(pred, target, valid, roi_side, px_scale):
    """Euclidean L1 in pixels (divided by px_scale), same unit as the metric."""
    err = torch.sqrt(((pred - target) ** 2).sum(-1) + 1e-8) * roi_side[:, None]
    mask = valid.float()

    return ((err / px_scale) * mask).sum() / mask.sum().clamp_min(1.0)


def heatmap_kl_loss(logits, target, valid):
    """KL(target distribution || softmax(logits)) per keypoint."""
    logp = F.log_softmax(logits.float().flatten(2), dim=-1)

    t = target.flatten(2)
    t = t / (t.sum(-1, keepdim=True) + 1e-8)

    kl = (t * (torch.log(t + 1e-8) - logp)).sum(-1)
    mask = valid.float()

    return (kl * mask).sum() / mask.sum().clamp_min(1.0)


def no_worse_loss(pred, initial, target, valid, roi_side, alpha, px_scale):
    """
    Hinge penalty (in pixels) on keypoints whose error exceeds alpha * initial
    error. The smaller the initial offset, the stricter the absolute requirement.
    """
    pred_err = torch.norm(pred - target, dim=-1) * roi_side[:, None]
    init_err = torch.norm(initial - target, dim=-1) * roi_side[:, None]

    hinge = F.relu(pred_err - alpha * init_err) / px_scale
    mask = valid.float()

    return (hinge * mask).sum() / mask.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# Epoch loop
# ---------------------------------------------------------------------------


def run_epoch(
    model, loader, optimizer, scaler, device, cfg, training, rigid_eval=False, desc=None
):
    model.train(training)

    t_cfg = cfg["training"]

    heatmap_weight = float(t_cfg["heatmap_loss_weight"])
    coordinate_weight = float(t_cfg["coordinate_loss_weight"])
    worse_weight = float(t_cfg.get("no_worse_loss_weight", 0.0))
    worse_alpha = float(t_cfg.get("no_worse_alpha", 1.0))
    px_scale = float(t_cfg.get("loss_px_scale", 16.0))
    stage_weights = t_cfg.get("stage_weights")

    use_amp = bool(t_cfg["use_amp"]) and device.type == "cuda"
    use_cond = bool(cfg["model"].get("use_size_condition", True))

    total_loss = 0.0
    total_heatmap_loss = 0.0
    total_coordinate_loss = 0.0
    total_worse_loss = 0.0

    total_pixel_error = 0.0
    total_stage1_error = 0.0
    total_initial_error = 0.0
    total_improvement = 0.0
    total_worse = 0.0
    total_pck3 = total_pck5 = total_pck10 = 0.0
    total_points = 0

    rigid_n = 0
    rigid_resid = 0.0
    rigid_proj_err = 0.0
    rigid_raw_err = 0.0
    rigid_gt_resid = 0.0

    progress = tqdm(
        loader, desc=desc or ("Train" if training else "Validation"), leave=False
    )

    for batch in progress:
        inputs = batch["input"].to(device, non_blocking=True)
        initial = batch["initial_coords"].to(device, non_blocking=True)
        target = batch["target_coords"].to(device, non_blocking=True)
        target_heatmaps = batch["target_heatmaps"].to(device, non_blocking=True)
        visibility = batch["visibility"].to(device, non_blocking=True)
        heatmap_valid = batch["heatmap_valid"].to(device, non_blocking=True)
        roi_side = batch["roi_side"].to(device, non_blocking=True)
        cond = batch["cond"].to(device, non_blocking=True) if use_cond else None

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                output = model(inputs, initial, cond)

            stages = output["stages"]

            if stage_weights and len(stage_weights) == len(stages):
                sw = [float(w) for w in stage_weights]
            else:
                sw = [1.0] * len(stages)

            sw_sum = sum(sw)

            coordinate_loss = (
                sum(
                    w
                    * coordinate_loss_px(
                        s["keypoints"].float(),
                        target,
                        heatmap_valid,
                        roi_side,
                        px_scale,
                    )
                    for w, s in zip(sw, stages)
                )
                / sw_sum
            )

            heatmap_loss = (
                sum(
                    w * heatmap_kl_loss(s["heatmaps"], target_heatmaps, heatmap_valid)
                    for w, s in zip(sw, stages)
                )
                / sw_sum
            )

            worse_loss = no_worse_loss(
                output["keypoints"].float(),
                initial,
                target,
                heatmap_valid,
                roi_side,
                worse_alpha,
                px_scale,
            )

            loss = (
                heatmap_weight * heatmap_loss
                + coordinate_weight * coordinate_loss
                + worse_weight * worse_loss
            )

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(t_cfg["grad_clip"])
                )

                scaler.step(optimizer)
                scaler.update()

        with torch.no_grad():
            pred_kp = output["keypoints"].float()
            stage1_kp = stages[0]["keypoints"].float()

            pred_err = torch.norm(pred_kp - target, dim=-1) * roi_side[:, None]
            stage1_err = torch.norm(stage1_kp - target, dim=-1) * roi_side[:, None]
            init_err = torch.norm(initial - target, dim=-1) * roi_side[:, None]

            valid_mask = (visibility > 0).float()
            improve = (init_err - pred_err) * valid_mask

            total_pixel_error += (pred_err * valid_mask).sum().item()
            total_stage1_error += (stage1_err * valid_mask).sum().item()
            total_initial_error += (init_err * valid_mask).sum().item()
            total_improvement += improve.sum().item()
            total_worse += ((pred_err > init_err).float() * valid_mask).sum().item()
            total_pck3 += ((pred_err < 3.0).float() * valid_mask).sum().item()
            total_pck5 += ((pred_err < 5.0).float() * valid_mask).sum().item()
            total_pck10 += ((pred_err < 10.0).float() * valid_mask).sum().item()
            total_points += valid_mask.sum().item()

            if rigid_eval and not training:
                pk = pred_kp.cpu().numpy().astype(np.float64)
                tg = target.cpu().numpy().astype(np.float64)
                rs = roi_side.cpu().numpy().astype(np.float64)
                bx = batch["roi_x0"].numpy().astype(np.float64)
                by = batch["roi_y0"].numpy().astype(np.float64)
                Ks = batch["K"].numpy().astype(np.float64)
                k3 = batch["kp3d"].numpy().astype(np.float64)
                vis = visibility.cpu().numpy()

                for b in range(pk.shape[0]):
                    if vis[b].min() <= 0:
                        continue

                    pred_px = np.stack(
                        [bx[b] + pk[b, :, 0] * rs[b], by[b] + pk[b, :, 1] * rs[b]], 1
                    )
                    gt_px = np.stack(
                        [bx[b] + tg[b, :, 0] * rs[b], by[b] + tg[b, :, 1] * rs[b]], 1
                    )

                    proj = rigid_project(pred_px, Ks[b], k3[b])
                    gt_proj = rigid_project(gt_px, Ks[b], k3[b])

                    if proj is None or gt_proj is None:
                        continue

                    rigid_n += 1
                    rigid_resid += np.linalg.norm(proj - pred_px, axis=1).mean()
                    rigid_proj_err += np.linalg.norm(proj - gt_px, axis=1).mean()
                    rigid_raw_err += np.linalg.norm(pred_px - gt_px, axis=1).mean()
                    rigid_gt_resid += np.linalg.norm(gt_proj - gt_px, axis=1).mean()

        total_loss += loss.item()
        total_heatmap_loss += heatmap_loss.item()
        total_coordinate_loss += coordinate_loss.item()
        total_worse_loss += worse_loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            coord=f"{coordinate_loss.item():.4f}",
            heat=f"{heatmap_loss.item():.4f}",
        )

    count = max(1, len(loader))
    points = max(1, total_points)
    nan = float("nan")

    return {
        "loss": total_loss / count,
        "heatmap_loss": total_heatmap_loss / count,
        "coordinate_loss": total_coordinate_loss / count,
        "no_worse_loss": total_worse_loss / count,
        "mean_pixel_error": total_pixel_error / points,
        "stage1_pixel_error": total_stage1_error / points,
        "initial_pixel_error": total_initial_error / points,
        "mean_improvement_px": total_improvement / points,
        "improvement_ratio": total_improvement / max(total_initial_error, 1e-9),
        "worse_ratio": total_worse / points,
        "pck3": total_pck3 / points,
        "pck5": total_pck5 / points,
        "pck10": total_pck10 / points,
        "rigid_samples": rigid_n,
        "rigid_residual_px": rigid_resid / rigid_n if rigid_n else nan,
        "rigid_raw_error_px": rigid_raw_err / rigid_n if rigid_n else nan,
        "rigid_projected_error_px": rigid_proj_err / rigid_n if rigid_n else nan,
        "rigid_gt_residual_px": rigid_gt_resid / rigid_n if rigid_n else nan,
    }


def format_metrics(tag, m):
    s = (
        f"{tag}: loss={m['loss']:.5f}, kl={m['heatmap_loss']:.4f}, "
        f"coord={m['coordinate_loss']:.4f}, nowors={m['no_worse_loss']:.4f} | "
        f"improve={m['mean_improvement_px']:+.3f}px "
        f"({m['improvement_ratio'] * 100:+.2f}%), "
        f"worse={m['worse_ratio'] * 100:.1f}% "
        f"[init {m['initial_pixel_error']:.2f} -> stage1 {m['stage1_pixel_error']:.2f} "
        f"-> final {m['mean_pixel_error']:.2f}px] "
        f"PCK3/5/10={m['pck3'] * 100:.1f}/{m['pck5'] * 100:.1f}/{m['pck10'] * 100:.1f}%"
    )

    if m["rigid_samples"]:
        s += (
            f"\n       rigid: residual={m['rigid_residual_px']:.3f}px "
            f"(GT {m['rigid_gt_residual_px']:.3f}px), "
            f"error raw {m['rigid_raw_error_px']:.3f} -> projected "
            f"{m['rigid_projected_error_px']:.3f}px (n={m['rigid_samples']})"
        )

    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = read_config(Path(args.config))

    set_seed(int(cfg["training"]["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    offset_cfg = build_offset_config(cfg)
    cur_cfg = merge_section(CURRICULUM_DEFAULTS, cfg.get("curriculum"))
    eval_cfg = merge_section(EVAL_DEFAULTS, cfg.get("eval"))

    print("Online offset config:")
    print(json.dumps(offset_cfg, indent=2))
    print("Curriculum config:")
    print(json.dumps(cur_cfg, indent=2))
    print("Eval config:")
    print(json.dumps(eval_cfg, indent=2))

    # New architecture: heatmaps are produced at stride 4 of the input.
    cfg["model"].setdefault("arch", "unet_softargmax")
    cfg["model"].setdefault("num_iters", 2)

    expected_hm = int(cfg["dataset"]["image_size"]) // 4

    if int(cfg["dataset"]["heatmap_size"]) != expected_hm:
        raise ValueError(
            f"heatmap_size must be image_size / 4 = {expected_hm}, "
            f"got {cfg['dataset']['heatmap_size']}"
        )

    samples = build_samples(cfg)

    train_samples, val_samples = split_samples(
        samples,
        float(cfg["dataset"]["train_ratio"]),
        int(cfg["dataset"]["split_seed"]),
    )

    print(f"Training samples  : {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")

    cur_enabled = bool(cur_cfg["enabled"])
    scale = float(cur_cfg["start_scale"]) if cur_enabled else 1.0

    train_dataset = KeypointDataset(
        train_samples,
        cfg,
        training=True,
        global_scale=scale,
        full_prob=float(cur_cfg["keep_full_prob"]) if cur_enabled else 0.0,
    )
    val_dataset = KeypointDataset(val_samples, cfg, training=False, global_scale=1.0)

    small_scale = float(eval_cfg["small_scale"])
    val_small_dataset = (
        KeypointDataset(val_samples, cfg, training=False, global_scale=small_scale)
        if small_scale > 0
        else None
    )

    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["training"]["num_workers"])

    # NOTE: keep persistent_workers=False so workers see the updated global_scale.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_dataset) > batch_size,
        persistent_workers=False,
    )

    def make_val_loader(ds):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            persistent_workers=False,
        )

    val_loader = make_val_loader(val_dataset)
    val_small_loader = make_val_loader(val_small_dataset) if val_small_dataset else None

    use_cond = bool(cfg["model"].get("use_size_condition", True))

    model = build_model_from_config(
        cfg["model"],
        cond_dim=COND_DIM if use_cond else 0,
        pretrained=bool(cfg["model"]["pretrained"]),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {n_params:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg["training"]["epochs"]), eta_min=1e-6
    )

    use_amp = bool(cfg["training"]["use_amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    save_dir = Path(cfg["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg_to_save = copy.deepcopy(cfg)
    cfg_to_save["offset"] = offset_cfg
    cfg_to_save["curriculum"] = cur_cfg
    cfg_to_save["eval"] = eval_cfg
    cfg_to_save["model"]["cond_dim"] = COND_DIM if use_cond else 0
    cfg_to_save["model"]["size_scale_mm"] = float(
        cfg["model"].get("size_scale_mm", 300.0)
    )

    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_to_save, f, indent=2, ensure_ascii=False)

    # The validation set is fixed, so absolute prediction error is comparable
    # across epochs even though the training offsets shrink. Select by it.
    best_val_error = float("inf")
    history = []
    epochs = int(cfg["training"]["epochs"])
    good_epochs = 0

    for epoch in range(1, epochs + 1):
        train_dataset.global_scale = scale

        print(f"\nEpoch [{epoch}/{epochs}]  curriculum_scale={scale:.3f}")

        train_metrics = run_epoch(
            model, train_loader, optimizer, scaler, device, cfg, training=True
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            scaler,
            device,
            cfg,
            training=False,
            rigid_eval=bool(eval_cfg["rigid"]),
            desc="Val(full)",
        )

        val_small_metrics = None

        if val_small_loader is not None:
            val_small_metrics = run_epoch(
                model,
                val_small_loader,
                optimizer,
                scaler,
                device,
                cfg,
                training=False,
                rigid_eval=bool(eval_cfg["rigid"]),
                desc="Val(small)",
            )

        scheduler.step()
        learning_rate = optimizer.param_groups[0]["lr"]

        print(format_metrics("Train      ", train_metrics))
        print(format_metrics("Valid full ", val_metrics))

        if val_small_metrics is not None:
            print(format_metrics("Valid small", val_small_metrics))

        print(f"Learning rate: {learning_rate:.8f}")

        # ---- curriculum controller (uses the training-set improvement) ----
        shrunk = False

        if cur_enabled:
            if train_metrics["improvement_ratio"] >= float(
                cur_cfg["trigger_improvement"]
            ):
                good_epochs += 1
            else:
                good_epochs = 0

            if good_epochs >= int(cur_cfg["patience"]) and scale > float(
                cur_cfg["min_scale"]
            ):
                new_scale = max(
                    float(cur_cfg["min_scale"]), scale * float(cur_cfg["shrink_factor"])
                )
                print(f"Curriculum: offset scale {scale:.3f} -> {new_scale:.3f}")
                scale = new_scale
                good_epochs = 0
                shrunk = True

        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "curriculum_scale": train_dataset.global_scale,
            "curriculum_shrunk": shrunk,
            "train": train_metrics,
            "validation": val_metrics,
        }

        if val_small_metrics is not None:
            record["validation_small"] = val_small_metrics

        history.append(record)

        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        current_error = val_metrics["mean_pixel_error"]

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "curriculum_scale": scale,
            "best_val_error": min(best_val_error, current_error),
            "val_metrics": val_metrics,
            "val_small_metrics": val_small_metrics,
            "config": cfg_to_save,
        }

        torch.save(checkpoint, save_dir / "last.pth")

        if current_error < best_val_error:
            best_val_error = current_error
            torch.save(checkpoint, save_dir / "best.pth")
            print(f"Saved best model: val pred error {best_val_error:.3f}px")

    print("\nTraining completed.")
    print(f"Best validation pixel error (full-range offsets): {best_val_error:.3f}px")
    print(f"Models saved to: {save_dir}")


if __name__ == "__main__":
    main()
