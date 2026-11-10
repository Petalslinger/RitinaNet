"""检查已转换数据集的图片、标注与统计文件是否正常。"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image


def check_split(root: Path, split: str, num_classes: int, strict: bool = False) -> dict:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    images = sorted(
        p for p in image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ) if image_dir.exists() else []
    if not images:
        print(f"[{split}] no images")
        return {"images": 0, "boxes": 0, "classes": {}, "empty_labels": 0, "bad_lines": 0}

    class_counts: Counter[int] = Counter()
    empty_labels = 0
    bad_lines = 0
    missing_labels = 0
    out_of_range = 0

    for idx, image_path in enumerate(images):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1
            continue
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            empty_labels += 1
            continue
        if idx < 200:
            with Image.open(image_path) as im:
                width, height = im.size
        else:
            width = height = None
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 5:
                bad_lines += 1
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:5])
            except ValueError:
                bad_lines += 1
                continue
            if cls < 0 or cls >= num_classes:
                out_of_range += 1
            class_counts[cls] += 1
            if width:
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    out_of_range += 1

    print(
        f"[{split}] images={len(images)} boxes={sum(class_counts.values())} "
        f"classes={dict(sorted(class_counts.items()))} "
        f"empty_labels={empty_labels} missing_labels={missing_labels} "
        f"bad_lines={bad_lines} out_of_range={out_of_range}"
    )

    if strict and (missing_labels or bad_lines or out_of_range):
        raise SystemExit(f"[{split}] strict check failed")

    return {
        "images": len(images),
        "boxes": int(sum(class_counts.values())),
        "classes": {str(k): int(v) for k, v in sorted(class_counts.items())},
        "empty_labels": empty_labels,
        "missing_labels": missing_labels,
        "bad_lines": bad_lines,
        "out_of_range": out_of_range,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/processed/cctsdb")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    yaml_path = root / "data.yaml"
    if not root.exists():
        raise SystemExit(f"dataset root not found: {root}")

    num_classes = 3
    names = ["mandatory", "prohibitory", "warning"]
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        num_classes = int(data.get("nc", 3))
        if isinstance(data.get("names"), dict):
            names = [data["names"][i] for i in sorted(data["names"])]
        elif isinstance(data.get("names"), list):
            names = data["names"]

    print(f"root={root.resolve()}")
    print(f"classes={names}")
    result = {}
    for split in ("train", "val", "test"):
        result[split] = check_split(root, split, num_classes, strict=args.strict)

    stats_path = root / "stats.json"
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            print("\nstats.json:")
            print(json.dumps(json.load(f), ensure_ascii=False, indent=2)[:2000])
    print("\ncheck done.")


if __name__ == "__main__":
    main()
