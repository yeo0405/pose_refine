#!/usr/bin/env python3
"""
BOP -> keypoint refinement dataset (no offset labels; offsets are generated online in train.py).

Output:
  output/
    images/<scene>/<id>.png
    labels/<scene>/<id>.txt          GT: obj_id cx cy w h (x y vis)*4   (normalized)
    meta/<scene>/<id>.json           per-image pose info for online offset generation
    models/<model_name>/points.ply   copied PLY files
    models/models_info.json          bbox size / center (mm) per obj_id
    classes.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from plyfile import PlyData


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_obj_id_from_dir(path: Path) -> Optional[int]:
    match = re.match(r"^(\d+)(?:_|$)", path.name)
    return int(match.group(1)) if match else None


def load_models(models_dir: Path, ply_unit: str) -> Dict[int, dict]:
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    models = {}

    for model_dir in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        obj_id = get_obj_id_from_dir(model_dir)

        if obj_id is None:
            print(f"[SKIP] Cannot get obj_id: {model_dir.name}")
            continue

        ply_files = sorted(model_dir.rglob("points.ply"))

        if not ply_files:
            print(f"[SKIP] points.ply not found: {model_dir}")
            continue

        if len(ply_files) > 1:
            print(f"[WARN] Multiple points.ply in {model_dir}; use {ply_files[0]}")

        ply_path = ply_files[0]
        ply = PlyData.read(str(ply_path))

        if "vertex" not in ply:
            raise ValueError(f"PLY has no vertex element: {ply_path}")

        data = ply["vertex"].data
        names = data.dtype.names or ()

        if not all(k in names for k in ("x", "y", "z")):
            raise ValueError(f"PLY must contain x, y, z: {ply_path}")

        points = np.column_stack((data["x"], data["y"], data["z"])).astype(np.float64)
        points = points[np.isfinite(points).all(axis=1)]

        if len(points) == 0:
            raise ValueError(f"PLY has no valid vertices: {ply_path}")

        if ply_unit == "m":
            points *= 1000.0

        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)

        models[obj_id] = {
            "bbox_min_mm": bbox_min,
            "bbox_max_mm": bbox_max,
            "size_mm": bbox_max - bbox_min,
            "offset_mm": (bbox_max + bbox_min) * 0.5,
            "ply_path": str(ply_path),
            "name": model_dir.name,
        }

        print(
            f"[MODEL] obj_id={obj_id} name={model_dir.name} "
            f"size_mm={np.round(models[obj_id]['size_mm'], 3).tolist()} "
            f"center_mm={np.round(models[obj_id]['offset_mm'], 3).tolist()}"
        )

    if not models:
        raise RuntimeError(f"No valid points.ply found under {models_dir}")

    return models


def export_models(models: Dict[int, dict], output_dir: Path, ply_unit: str):
    """Copy PLY files into output/models and write models_info.json (all in mm)."""
    out_models = output_dir / "models"
    out_models.mkdir(parents=True, exist_ok=True)

    info = {}

    for obj_id, m in sorted(models.items()):
        dst_dir = out_models / m["name"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_ply = dst_dir / "points.ply"
        shutil.copy2(m["ply_path"], dst_ply)

        info[str(obj_id)] = {
            "name": m["name"],
            "ply": str(dst_ply.relative_to(output_dir)),
            "ply_unit": ply_unit,  # unit of the copied PLY file itself
            "size_mm": m["size_mm"].tolist(),
            "center_mm": m["offset_mm"].tolist(),
            "bbox_min_mm": m["bbox_min_mm"].tolist(),
            "bbox_max_mm": m["bbox_max_mm"].tolist(),
        }

    with (out_models / "models_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def face_corners_obj(box: dict, sign: int) -> np.ndarray:
    hx, hy, hz = box["size_mm"] / 2.0

    corners = np.array(
        [
            [-hx, -hy, sign * hz],
            [+hx, -hy, sign * hz],
            [+hx, +hy, sign * hz],
            [-hx, +hy, sign * hz],
        ],
        dtype=np.float64,
    )

    return corners + box["offset_mm"]


def pick_visible_face(box: dict, R: np.ndarray, t: np.ndarray):
    best = None

    for sign in (+1, -1):
        normal = R @ np.array([0.0, 0.0, float(sign)], dtype=np.float64)

        center_obj = box["offset_mm"] + np.array(
            [0.0, 0.0, sign * box["size_mm"][2] / 2.0], dtype=np.float64
        )

        center_cam = R @ center_obj + t
        cosine = float(-normal @ center_cam / (np.linalg.norm(center_cam) + 1e-9))

        if best is None or cosine > best[2]:
            best = (sign, face_corners_obj(box, sign), cosine)

    return best


def project_points(
    points_obj: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    points_cam = (R @ points_obj.T).T + t.reshape(1, 3)

    z = points_cam[:, 2]
    valid = z > 1e-6

    points_2d = np.full((len(points_obj), 2), np.nan, dtype=np.float64)

    if np.any(valid):
        projected = (K @ points_cam[valid].T).T
        points_2d[valid] = projected[:, :2] / projected[:, 2:3]

    return points_2d, valid


def load_mask(path: Path) -> Optional[np.ndarray]:
    if not path.is_file():
        return None

    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def mask_bbox(mask: Optional[np.ndarray]) -> Optional[Tuple[int, int, int, int]]:
    if mask is None:
        return None

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def projected_bbox(box, R, t, K, width, height):
    xmin, ymin, zmin = box["bbox_min_mm"]
    xmax, ymax, zmax = box["bbox_max_mm"]

    corners = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        dtype=np.float64,
    )

    points_2d, valid = project_points(corners, R, t, K)
    points_2d = points_2d[valid]

    if len(points_2d) == 0:
        return None

    x1 = max(0, int(np.floor(np.min(points_2d[:, 0]))))
    y1 = max(0, int(np.floor(np.min(points_2d[:, 1]))))
    x2 = min(width - 1, int(np.ceil(np.max(points_2d[:, 0]))))
    y2 = min(height - 1, int(np.ceil(np.max(points_2d[:, 1]))))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def get_yolo_bbox(mask, fallback_mask, box, R, t, K, width, height):
    bounds = mask_bbox(mask)

    if bounds is None:
        bounds = mask_bbox(fallback_mask)

    if bounds is None:
        bounds = projected_bbox(box, R, t, K, width, height)

    if bounds is None:
        return None

    x1, y1, x2, y2 = bounds

    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)

    xc = (x1 + x2 + 1) * 0.5 / width
    yc = (y1 + y2 + 1) * 0.5 / height

    return (
        float(np.clip(xc, 0.0, 1.0)),
        float(np.clip(yc, 0.0, 1.0)),
        float(np.clip(bw / width, 0.0, 1.0)),
        float(np.clip(bh / height, 0.0, 1.0)),
    )


def point_in_mask(mask, x, y, radius) -> bool:
    if mask is None or not np.isfinite(x) or not np.isfinite(y):
        return False

    h, w = mask.shape[:2]
    ix, iy = int(round(x)), int(round(y))

    x1, x2 = max(0, ix - radius), min(w, ix + radius + 1)
    y1, y2 = max(0, iy - radius), min(h, iy + radius + 1)

    if x1 >= x2 or y1 >= y2:
        return False

    return bool(np.any(mask[y1:y2, x1:x2] > 0))


def keypoint_visibility(
    points_2d, valid_z, width, height, mask_visib, radius
) -> List[int]:
    visibility = []

    for i, (x, y) in enumerate(points_2d):
        if not valid_z[i] or not np.isfinite(x) or not np.isfinite(y):
            visibility.append(0)
            continue

        if not (0.0 <= x < width and 0.0 <= y < height):
            visibility.append(0)
            continue

        if mask_visib is None:
            visibility.append(2)
        else:
            visibility.append(2 if point_in_mask(mask_visib, x, y, radius) else 1)

    return visibility


def normalized_keypoints(points_2d, visibility, width, height) -> List[float]:
    values = []

    for (x, y), v in zip(points_2d, visibility):
        if np.isfinite(x) and np.isfinite(y):
            xn = float(np.clip(x / width, 0.0, 1.0))
            yn = float(np.clip(y / height, 0.0, 1.0))
        else:
            xn, yn = 0.0, 0.0

        values.extend((xn, yn, int(v)))

    return values


def format_label(label_id, bbox, keypoint_values) -> str:
    values = [label_id, *bbox, *keypoint_values]
    return " ".join(str(v) if isinstance(v, int) else f"{v:.6f}" for v in values)


def find_rgb(rgb_dir: Path, image_id: int) -> Optional[Path]:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        p = rgb_dir / f"{image_id:06d}{suffix}"

        if p.is_file():
            return p

    return None


def process_scene(
    scene_dir: Path,
    models: Dict[int, dict],
    selected_ids: set,
    output_dir: Path,
    visibility_radius: int,
    translation_scale: float,
) -> int:
    rgb_dir = scene_dir / "rgb"
    gt_path = scene_dir / "scene_gt.json"
    camera_path = scene_dir / "scene_camera.json"
    mask_dir = scene_dir / "mask"
    mask_visib_dir = scene_dir / "mask_visib"

    if not rgb_dir.is_dir() or not gt_path.is_file() or not camera_path.is_file():
        print(f"[SKIP] Invalid BOP scene: {scene_dir}")
        return 0

    scene_gt = load_json(gt_path)
    scene_camera = load_json(camera_path)

    image_out_dir = output_dir / "images" / scene_dir.name
    label_out_dir = output_dir / "labels" / scene_dir.name
    meta_out_dir = output_dir / "meta" / scene_dir.name

    total_instances = 0

    for image_key in sorted(scene_gt.keys(), key=lambda x: int(x)):
        image_id = int(image_key)

        if image_key not in scene_camera:
            continue

        rgb_path = find_rgb(rgb_dir, image_id)

        if rgb_path is None:
            print(f"[WARN] RGB not found: scene={scene_dir.name}, image={image_id}")
            continue

        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)

        if image is None:
            print(f"[WARN] Cannot read image: {rgb_path}")
            continue

        height, width = image.shape[:2]

        K = np.asarray(scene_camera[image_key]["cam_K"], dtype=np.float64).reshape(3, 3)

        label_lines = []
        meta_instances = []

        for gt_index, gt in enumerate(scene_gt[image_key]):
            obj_id = int(gt["obj_id"])

            if obj_id not in selected_ids or obj_id not in models:
                continue

            box = models[obj_id]

            R = np.asarray(gt["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
            t = np.asarray(gt["cam_t_m2c"], dtype=np.float64).reshape(3)
            t = t * translation_scale  # -> mm

            sign, keypoints_3d, face_cos = pick_visible_face(box, R, t)
            keypoints_2d, valid_z = project_points(keypoints_3d, R, t, K)

            full_mask = load_mask(mask_dir / f"{image_id:06d}_{gt_index:06d}.png")
            visible_mask = load_mask(
                mask_visib_dir / f"{image_id:06d}_{gt_index:06d}.png"
            )

            bbox = get_yolo_bbox(full_mask, visible_mask, box, R, t, K, width, height)

            if bbox is None:
                print(
                    f"[WARN] Cannot determine bbox: scene={scene_dir.name} "
                    f"image={image_id} obj_id={obj_id} gt={gt_index}"
                )
                continue

            visibility = keypoint_visibility(
                keypoints_2d, valid_z, width, height, visible_mask, visibility_radius
            )

            kp_values = normalized_keypoints(keypoints_2d, visibility, width, height)

            label_lines.append(format_label(obj_id, bbox, kp_values))

            # Pose info for online offset generation (object frame, mm)
            meta_instances.append(
                {
                    "obj_id": obj_id,
                    "gt_index": gt_index,
                    "face_sign": int(sign),
                    "face_cosine": float(face_cos),
                    "R_m2c": R.reshape(-1).tolist(),
                    "t_m2c_mm": t.tolist(),
                    "keypoints_3d_mm": keypoints_3d.tolist(),
                    "model_center_mm": box["offset_mm"].tolist(),
                    "model_size_mm": box["size_mm"].tolist(),
                }
            )

            total_instances += 1

        if not label_lines:
            continue

        for d in (image_out_dir, label_out_dir, meta_out_dir):
            d.mkdir(parents=True, exist_ok=True)

        shutil.copy2(rgb_path, image_out_dir / rgb_path.name)

        (label_out_dir / f"{rgb_path.stem}.txt").write_text(
            "\n".join(label_lines) + "\n", encoding="utf-8"
        )

        meta = {
            "image": f"{scene_dir.name}/{rgb_path.name}",
            "width": int(width),
            "height": int(height),
            "cam_K": K.reshape(-1).tolist(),
            "units": "mm",
            "instances": meta_instances,
        }

        with (meta_out_dir / f"{rgb_path.stem}.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f)

    return total_instances


def main():
    parser = argparse.ArgumentParser(
        description="Convert BOP data to keypoint-refinement data (GT + pose meta + PLY)."
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="BOP train_pbr directory"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Output dataset directory"
    )
    parser.add_argument(
        "--models-dir",
        required=True,
        type=Path,
        help="Root directory containing numbered model folders",
    )
    parser.add_argument("--ply-unit", choices=("m", "mm"), default="m")
    parser.add_argument("--translation-unit", choices=("mm", "m"), default="mm")
    parser.add_argument("--obj-ids", nargs="*", type=int, default=None)
    parser.add_argument("--visibility-radius", type=int, default=3)

    args = parser.parse_args()

    input_dir = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    models_dir = args.models_dir.expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"BOP input directory not found: {input_dir}")

    models = load_models(models_dir, args.ply_unit)

    selected_ids = set(args.obj_ids) if args.obj_ids else set(models.keys())
    missing = selected_ids - set(models.keys())

    if missing:
        raise ValueError(f"No PLY model found for obj_ids: {sorted(missing)}")

    translation_scale = 1000.0 if args.translation_unit == "m" else 1.0

    output_dir.mkdir(parents=True, exist_ok=True)

    export_models(
        {k: v for k, v in models.items() if k in selected_ids},
        output_dir,
        args.ply_unit,
    )

    classes = {
        "id_type": "bop_obj_id",
        "obj_ids": {str(i): models[i]["name"] for i in sorted(selected_ids)},
        "kpt_shape": [4, 3],
        "label_format": "obj_id cx cy w h (x y visibility) * 4",
        "meta_format": "meta/<scene>/<id>.json: cam_K, instances[R_m2c, t_m2c_mm, keypoints_3d_mm, model_center_mm]",
        "meta_units": "mm",
    }

    with (output_dir / "classes.json").open("w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2, ensure_ascii=False)

    scene_dirs = sorted(
        p for p in input_dir.iterdir() if p.is_dir() and p.name.isdigit()
    )

    if not scene_dirs:
        raise RuntimeError(f"No numeric scene folders found: {input_dir}")

    total = 0

    for scene_dir in scene_dirs:
        print(f"\n[SCENE] {scene_dir.name}")
        count = process_scene(
            scene_dir,
            models,
            selected_ids,
            output_dir,
            args.visibility_radius,
            translation_scale,
        )
        total += count
        print(f"  instances: {count}")

    print("\n" + "=" * 55)
    print("[DONE] Dataset generation completed")
    print("=" * 55)
    print(f"[DONE] Instances: {total}")
    print(f"[DONE] Images : {output_dir / 'images'}")
    print(f"[DONE] GT     : {output_dir / 'labels'}")
    print(f"[DONE] Meta   : {output_dir / 'meta'}")
    print(f"[DONE] Models : {output_dir / 'models'}")


if __name__ == "__main__":
    main()
