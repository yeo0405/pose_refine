#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomly visualize BOP keypoint labels."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset root containing images/ and labels/.",
    )

    parser.add_argument(
        "--label-type",
        choices=("gt", "offset"),
        default="gt",
        help="Select original GT or rigid-body offset keypoints.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Visualization output directory.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=30,
        help="Number of images to sample.",
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--obj-ids",
        type=int,
        nargs="*",
        default=None,
        help="Only visualize these BOP obj_ids.",
    )

    parser.add_argument("--point-radius", type=int, default=6)
    parser.add_argument("--font-scale", type=float, default=0.55)
    parser.add_argument("--thickness", type=int, default=2)

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each sampled result.",
    )

    parser.add_argument(
        "--wait",
        type=int,
        default=0,
        help="Display wait time in ms; 0 waits for a key.",
    )

    return parser.parse_args()


def load_class_names(dataset_root: Path):
    path = dataset_root / "classes.json"

    if not path.is_file():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Cannot read {path}: {exc}")
        return {}

    mapping = data.get("obj_ids", {})

    if isinstance(mapping, dict):
        return {int(k): str(v) for k, v in mapping.items()}

    return {}


def parse_label_line(
    line: str,
    label_path: Path,
    line_number: int,
):
    values = line.strip().split()

    if not values:
        return None

    try:
        numbers = [float(v) for v in values]
    except ValueError as exc:
        raise ValueError(
            f"Invalid numeric value in {label_path}:{line_number}"
        ) from exc

    if len(numbers) < 17:
        raise ValueError(
            f"Expected at least 17 values, got {len(numbers)} "
            f"in {label_path}:{line_number}"
        )

    obj_id_float = numbers[0]
    obj_id = int(round(obj_id_float))

    if abs(obj_id_float - obj_id) > 1e-5:
        raise ValueError(f"obj_id must be an integer in " f"{label_path}:{line_number}")

    bbox = numbers[1:5]

    keypoints = np.asarray(
        numbers[5:17],
        dtype=np.float32,
    ).reshape(4, 3)

    return {
        "obj_id": obj_id,
        "bbox": bbox,
        "keypoints": keypoints,
    }


def draw_instance(
    image,
    instance,
    name,
    radius,
    font_scale,
    thickness,
):
    height, width = image.shape[:2]

    obj_id = instance["obj_id"]
    cx, cy, bw, bh = instance["bbox"]

    x1 = int(round((cx - bw / 2.0) * width))
    y1 = int(round((cy - bh / 2.0) * height))

    x2 = int(round((cx + bw / 2.0) * width))
    y2 = int(round((cy + bh / 2.0) * height))

    cv2.rectangle(
        image,
        (max(0, x1), max(0, y1)),
        (min(width - 1, x2), min(height - 1, y2)),
        (255, 180, 0),
        thickness,
    )

    title = f"obj_id={obj_id}"

    if name:
        title += f" ({name})"

    text_y = max(18, min(height - 4, y1 - 7))

    cv2.putText(
        image,
        title,
        (max(2, x1), text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 180, 0),
        thickness,
        cv2.LINE_AA,
    )

    colors = {
        0: (0, 0, 255),
        1: (0, 165, 255),
        2: (0, 220, 0),
    }

    for index, (kx, ky, visibility) in enumerate(instance["keypoints"]):
        v = int(round(float(visibility)))

        color = colors.get(v, (255, 0, 255))

        px = int(round(float(kx) * width))
        py = int(round(float(ky) * height))

        if v == 0:
            arm = radius + 2

            cv2.line(
                image,
                (px - arm, py - arm),
                (px + arm, py + arm),
                color,
                thickness,
                cv2.LINE_AA,
            )

            cv2.line(
                image,
                (px - arm, py + arm),
                (px + arm, py - arm),
                color,
                thickness,
                cv2.LINE_AA,
            )

        else:
            cv2.circle(
                image,
                (px, py),
                radius,
                color,
                -1,
                cv2.LINE_AA,
            )

            cv2.circle(
                image,
                (px, py),
                radius + 2,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        label = f"{index}:v{v}"

        tx = min(
            max(0, px + radius + 3),
            max(0, width - 65),
        )

        ty = min(
            max(14, py - radius - 3),
            max(14, height - 3),
        )

        cv2.putText(
            image,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def find_image(images_root: Path, relative_path: Path):
    image_path = (images_root / relative_path).with_suffix(".png")

    if image_path.is_file():
        return image_path

    for ext in (
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    ):
        candidate = (images_root / relative_path).with_suffix(ext)

        if candidate.is_file():
            return candidate

    return None


def main():
    args = parse_args()

    dataset_root = Path(args.dataset).expanduser().resolve()

    images_root = dataset_root / "images"

    label_folder = "labels" if args.label_type == "gt" else "labels_offset"

    labels_root = dataset_root / label_folder

    if not images_root.is_dir() or not labels_root.is_dir():
        raise FileNotFoundError(
            f"Expected both directories:\n" f"  {images_root}\n" f"  {labels_root}"
        )

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be greater than 0")

    output_root = (
        Path(args.output).expanduser().resolve()
        if args.output
        else dataset_root / f"visualized_{args.label_type}"
    )

    class_names = load_class_names(dataset_root)

    allowed_ids = set(args.obj_ids) if args.obj_ids else None

    samples = []

    for label_path in sorted(labels_root.rglob("*.txt")):
        relative_path = label_path.relative_to(labels_root)

        image_path = find_image(
            images_root,
            relative_path,
        )

        if image_path is None:
            continue

        instances = []

        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(),
            1,
        ):
            if not line.strip():
                continue

            item = parse_label_line(
                line,
                label_path,
                line_number,
            )

            if item is not None and (
                allowed_ids is None or item["obj_id"] in allowed_ids
            ):
                instances.append(item)

        if instances:
            samples.append((image_path, label_path, instances))

    if not samples:
        raise RuntimeError("No matching image/label pairs found.")

    rng = random.Random(args.seed)

    selected = rng.sample(
        samples,
        min(args.num_samples, len(samples)),
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"[INFO] Label type: {args.label_type}")
    print(f"[INFO] Labels: {labels_root}")
    print(f"[INFO] Found {len(samples)} images.")
    print(f"[INFO] Sampling {len(selected)} images " f"(seed={args.seed}).")
    print(f"[INFO] Output: {output_root}")

    for index, (
        image_path,
        label_path,
        instances,
    ) in enumerate(selected, 1):

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            print(f"[WARN] Cannot read image: {image_path}")
            continue

        for instance in instances:
            draw_instance(
                image,
                instance,
                class_names.get(
                    instance["obj_id"],
                    "",
                ),
                args.point_radius,
                args.font_scale,
                args.thickness,
            )

        header = (
            f"{args.label_type.upper()} | "
            f"sample {index}/{len(selected)} | "
            f"{image_path.parent.name}/{image_path.name}"
        )

        cv2.putText(
            image,
            header,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        relative_path = image_path.relative_to(images_root)

        save_path = (output_root / relative_path).with_suffix(".jpg")

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not cv2.imwrite(
            str(save_path),
            image,
        ):
            print(f"[WARN] Failed to save: {save_path}")
            continue

        print(f"[{index:03d}/{len(selected):03d}] " f"{save_path}")

        if args.show:
            cv2.imshow(
                f"Keypoints - {args.label_type}",
                image,
            )

            key = cv2.waitKey(args.wait) & 0xFF

            if key == 27 or key == ord("q"):
                break

    if args.show:
        cv2.destroyAllWindows()

    print("[DONE]")


if __name__ == "__main__":
    main()
