"""按每图目标数量抽取密集子集，用于验证 Soft-NMS 在密集场景下的收益。"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

SPLITS = ("train", "val", "test")
IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def read_class_names(root: Path) -> list[str]:
    """优先读 classes.txt，其次读 data.yaml 的 names。"""
    classes_file = root / "classes.txt"
    if classes_file.is_file():
        names = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines()]
        names = [name for name in names if name]
        if names:
            return names

    yaml_file = root / "data.yaml"
    if yaml_file.is_file():
        try:
            import yaml

            data = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}
            names = data.get("names")
            if isinstance(names, dict):
                return [str(names[key]) for key in sorted(names)]
            if isinstance(names, list):
                return [str(name) for name in names]
        except Exception:
            pass
    return []


def count_boxes(label_path: Path) -> tuple[int, list[int]]:
    """返回 (框数, 类别列表)；坏行忽略。"""
    labels: list[int] = []
    try:
        text = label_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0, labels
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            labels.append(int(float(parts[0])))
        except ValueError:
            continue
    return len(labels), labels


def collect(root: Path, splits: Iterable[str]) -> dict[str, list[dict]]:
    """每个 split 收集 {image, label, count, labels}。"""
    items: dict[str, list[dict]] = {}
    for split in splits:
        label_dir = root / "labels" / split
        if not label_dir.is_dir():
            continue
        entries: list[dict] = []
        for label_path in sorted(label_dir.glob("*.txt")):
            count, labels = count_boxes(label_path)
            if count == 0:
                continue
            image_path = None
            for suffix in IMG_SUFFIXES:
                candidate = root / "images" / split / f"{label_path.stem}{suffix}"
                if candidate.is_file():
                    image_path = candidate
                    break
            if image_path is None:
                continue
            entries.append(
                {"image": image_path, "label": label_path, "count": count, "labels": labels}
            )
        items[split] = entries
    return items


def histogram(counts: list[int]) -> str:
    """框数分布，按 1/2/3-4/5-8/9+ 分桶。"""
    if not counts:
        return "no images"
    buckets = Counter()
    for count in counts:
        if count == 1:
            buckets["1"] += 1
        elif count == 2:
            buckets["2"] += 1
        elif count <= 4:
            buckets["3-4"] += 1
        elif count <= 8:
            buckets["5-8"] += 1
        else:
            buckets["9+"] += 1
    total = len(counts)
    mean = sum(counts) / total
    parts = [
        f"{key}: {buckets[key]} ({buckets[key] / total * 100:.1f}%)"
        for key in ["1", "2", "3-4", "5-8", "9+"]
    ]
    return f"total: {total} images, mean {mean:.2f} boxes/image | " + " | ".join(parts)


def subset(entries: list[dict], min_boxes: int, max_boxes: int | None) -> list[dict]:
    selected = [entry for entry in entries if entry["count"] >= min_boxes]
    if max_boxes is not None:
        selected = [entry for entry in selected if entry["count"] <= max_boxes]
    return selected


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def export(
    selected: dict[str, list[dict]], out_dir: Path, class_names: list[str], mode: str
) -> None:
    for split, entries in selected.items():
        for entry in entries:
            link_or_copy(entry["image"], out_dir / "images" / split / entry["image"].name, mode)
            link_or_copy(entry["label"], out_dir / "labels" / split / entry["label"].name, mode)

    (out_dir / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")

    names = ", ".join(f'"{name}"' for name in class_names)
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(class_names)}\n"
        f"names: [{names}]\n",
        encoding="utf-8",
    )

    stats: dict[str, dict] = {}
    for split, entries in selected.items():
        counts = [entry["count"] for entry in entries]
        class_counts: Counter = Counter()
        for entry in entries:
            class_counts.update(entry["labels"])
        stats[split] = {
            "images": len(entries),
            "boxes": sum(counts),
            "mean_boxes": (sum(counts) / len(counts)) if counts else 0.0,
            "class_counts": {str(key): value for key, value in sorted(class_counts.items())},
        }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/processed/cctsdb", help="已转换好的数据集目录")
    parser.add_argument("--out", default="data/processed/cctsdb_dense3", help="输出目录")
    parser.add_argument("--min-boxes", type=int, default=3, help="每图最少框数，>= 该值的图入选")
    parser.add_argument("--max-boxes", type=int, default=None, help="每图最多框数（默认不限）")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--stats-only", action="store_true", help="只打印分布，不写文件")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"dataset root not found: {root}")

    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    class_names = read_class_names(root)
    items = collect(root, splits)
    if not items:
        raise SystemExit(f"no labels found under {root / 'labels'}")

    for split, entries in items.items():
        counts = [entry["count"] for entry in entries]
        print(f"[dense] {split}: {histogram(counts)}")
    print(f"[dense] classes: {class_names}")

    selected = {
        split: subset(entries, args.min_boxes, args.max_boxes) for split, entries in items.items()
    }
    for split, entries in selected.items():
        print(f"[dense] {split}: {len(entries)} images with >= {args.min_boxes} boxes")

    if args.stats_only:
        return

    if not selected.get("train"):
        raise SystemExit("no training images matched --min-boxes, lower it and retry")

    out_dir = Path(args.out)
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_dir} exists, pass --overwrite to replace it")
        shutil.rmtree(out_dir)
    for split in splits:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    export(selected, out_dir, class_names, args.copy_mode)
    print(f"[dense] written to {out_dir} (mode={args.copy_mode})")


if __name__ == "__main__":
    main()
