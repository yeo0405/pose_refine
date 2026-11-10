#!/usr/bin/env python3
"""Merge multiple BOP keypoint-refinement datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def resolve_ply_path(dataset_dir: Path, ply_value: str) -> Path:
    """Resolve PLY paths stored in models_info.json."""

    relative_path = Path(ply_value)

    candidates = [
        dataset_dir / relative_path,
        dataset_dir / "models" / relative_path,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"PLY not found:\n"
        f"  Dataset: {dataset_dir}\n"
        f"  Relative path: {ply_value}\n"
        f"  Checked:\n" + "\n".join(f"    {p}" for p in candidates)
    )


def get_scenes(dataset_dir: Path):
    image_dir = dataset_dir / "images"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {image_dir}")

    return sorted(p for p in image_dir.iterdir() if p.is_dir())


def load_dataset_models(dataset_dir: Path):
    info_path = dataset_dir / "models" / "models_info.json"
    classes_path = dataset_dir / "classes.json"

    if not info_path.is_file():
        raise FileNotFoundError(f"models_info.json not found: {info_path}")

    if not classes_path.is_file():
        raise FileNotFoundError(f"classes.json not found: {classes_path}")

    models_info = load_json(info_path)
    classes = load_json(classes_path)

    models = {}

    for obj_id, info in models_info.items():
        obj_id = str(obj_id)

        ply_path = resolve_ply_path(dataset_dir, info["ply"])

        models[obj_id] = {
            "info": info,
            "ply_path": ply_path,
            "hash": file_hash(ply_path),
        }

    class_ids = set(classes.get("obj_ids", {}).keys())
    model_ids = set(models.keys())

    if class_ids != model_ids:
        raise ValueError(
            f"Class/model ID mismatch in {dataset_dir}:\n"
            f"  Classes: {sorted(class_ids)}\n"
            f"  Models: {sorted(model_ids)}"
        )

    for obj_id in class_ids:
        class_name = classes["obj_ids"][obj_id]
        model_name = models[obj_id]["info"]["name"]

        if class_name != model_name:
            raise ValueError(
                f"Class/model name mismatch:\n"
                f"  obj_id: {obj_id}\n"
                f"  class: {class_name}\n"
                f"  model: {model_name}"
            )

    return models, classes


def validate_models(
    all_models: Dict[str, dict],
    new_models: Dict[str, dict],
):
    for obj_id, model in new_models.items():

        if obj_id not in all_models:
            all_models[obj_id] = model
            continue

        old = all_models[obj_id]

        old_info = old["info"]
        new_info = model["info"]

        if old_info["name"] != new_info["name"]:
            raise ValueError(
                f"obj_id={obj_id} has different model names:\n"
                f"  {old_info['name']}\n"
                f"  {new_info['name']}"
            )

        if old["hash"] != model["hash"]:
            raise ValueError(
                f"obj_id={obj_id} has different PLY files:\n"
                f"  {old['ply_path']}\n"
                f"  {model['ply_path']}\n"
                "Cannot merge datasets with conflicting models."
            )

        for key in (
            "size_mm",
            "center_mm",
            "bbox_min_mm",
            "bbox_max_mm",
        ):
            if key in old_info and key in new_info:
                if old_info[key] != new_info[key]:
                    raise ValueError(
                        f"Model metadata mismatch:\n"
                        f"  obj_id: {obj_id}\n"
                        f"  field: {key}"
                    )


def export_models(
    all_models: Dict[str, dict],
    output_dir: Path,
):
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    merged_info = {}

    for obj_id, model in sorted(
        all_models.items(),
        key=lambda x: int(x[0]),
    ):
        info = dict(model["info"])
        model_name = info["name"]

        dst_dir = models_dir / model_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst_ply = dst_dir / "points.ply"

        if not dst_ply.exists():
            shutil.copy2(model["ply_path"], dst_ply)

        info["ply"] = str(dst_ply.relative_to(output_dir))

        merged_info[obj_id] = info

    save_json(
        models_dir / "models_info.json",
        merged_info,
    )


def merge_scene(
    dataset_dir: Path,
    scene_dir: Path,
    output_dir: Path,
    new_scene_id: str,
):
    old_scene_id = scene_dir.name

    src_image_dir = dataset_dir / "images" / old_scene_id
    src_label_dir = dataset_dir / "labels" / old_scene_id
    src_meta_dir = dataset_dir / "meta" / old_scene_id

    dst_image_dir = output_dir / "images" / new_scene_id
    dst_label_dir = output_dir / "labels" / new_scene_id
    dst_meta_dir = output_dir / "meta" / new_scene_id

    if not src_label_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {src_label_dir}")

    if not src_meta_dir.is_dir():
        raise FileNotFoundError(f"Meta directory not found: {src_meta_dir}")

    dst_image_dir.mkdir(parents=True, exist_ok=True)
    dst_label_dir.mkdir(parents=True, exist_ok=True)
    dst_meta_dir.mkdir(parents=True, exist_ok=True)

    image_count = 0

    for image_path in sorted(src_image_dir.iterdir()):

        if not image_path.is_file():
            continue

        label_path = src_label_dir / f"{image_path.stem}.txt"
        meta_path = src_meta_dir / f"{image_path.stem}.json"

        if not label_path.is_file():
            raise FileNotFoundError(f"Label not found: {label_path}")

        if not meta_path.is_file():
            raise FileNotFoundError(f"Meta not found: {meta_path}")

        dst_image_path = dst_image_dir / image_path.name
        dst_label_path = dst_label_dir / label_path.name
        dst_meta_path = dst_meta_dir / meta_path.name

        shutil.copy2(image_path, dst_image_path)
        shutil.copy2(label_path, dst_label_path)

        meta = load_json(meta_path)

        meta["image"] = f"{new_scene_id}/{image_path.name}"

        save_json(dst_meta_path, meta)

        image_count += 1

    return image_count


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple keypoint-refinement datasets."
    )

    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        type=Path,
        help="Input dataset directories.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Merged dataset output directory.",
    )

    parser.add_argument(
        "--scene-start",
        type=int,
        default=0,
        help="Starting scene ID. Default: 0.",
    )

    args = parser.parse_args()

    input_dirs = [p.expanduser().resolve() for p in args.inputs]

    output_dir = args.output.expanduser().resolve()

    if output_dir in input_dirs:
        raise ValueError("Output directory cannot be an input dataset.")

    for dataset_dir in input_dirs:
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}\n"
            "Please use a new output directory."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    all_models = {}
    merged_classes = None

    # Validate models and classes.
    for dataset_dir in input_dirs:

        print(f"\n[DATASET] {dataset_dir}")

        models, classes = load_dataset_models(dataset_dir)

        validate_models(all_models, models)

        if merged_classes is None:
            merged_classes = dict(classes)
            merged_classes["obj_ids"] = dict(classes["obj_ids"])

        else:
            for obj_id, name in classes["obj_ids"].items():

                old_name = merged_classes["obj_ids"].get(obj_id)

                if old_name is not None and old_name != name:
                    raise ValueError(
                        f"Class name conflict:\n"
                        f"  obj_id: {obj_id}\n"
                        f"  {old_name} vs {name}"
                    )

                merged_classes["obj_ids"][obj_id] = name

        print(f"  Models: {len(models)}")

    # Export merged models.
    export_models(all_models, output_dir)

    merged_classes["obj_ids"] = {
        obj_id: name
        for obj_id, name in sorted(
            merged_classes["obj_ids"].items(),
            key=lambda x: int(x[0]),
        )
    }

    save_json(
        output_dir / "classes.json",
        merged_classes,
    )

    # Merge scenes.
    next_scene_id = args.scene_start

    total_images = 0
    total_scenes = 0

    for dataset_index, dataset_dir in enumerate(input_dirs):

        scenes = get_scenes(dataset_dir)

        print(f"\n[MERGE] Dataset " f"{dataset_index + 1}/{len(input_dirs)}")

        print(f"  Source: {dataset_dir}")
        print(f"  Scenes: {len(scenes)}")

        for scene_dir in scenes:

            new_scene_id = f"{next_scene_id:06d}"

            count = merge_scene(
                dataset_dir,
                scene_dir,
                output_dir,
                new_scene_id,
            )

            print(f"  {scene_dir.name} -> " f"{new_scene_id} ({count} images)")

            total_images += count
            total_scenes += 1
            next_scene_id += 1

    print("\n" + "=" * 60)
    print("[DONE] Dataset merge completed")
    print("=" * 60)

    print(f"[DONE] Datasets : {len(input_dirs)}")
    print(f"[DONE] Scenes   : {total_scenes}")
    print(f"[DONE] Images   : {total_images}")
    print(f"[DONE] Models   : {len(all_models)}")

    print(f"[DONE] Output   : {output_dir}")
    print(f"[DONE] Images   : {output_dir / 'images'}")
    print(f"[DONE] Labels   : {output_dir / 'labels'}")
    print(f"[DONE] Meta     : {output_dir / 'meta'}")
    print(f"[DONE] Models   : {output_dir / 'models'}")


if __name__ == "__main__":
    main()
