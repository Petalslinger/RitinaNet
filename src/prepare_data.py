"""把 VOC / polygon 标注转换成 YOLO 格式并划分数据集。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from .download_data import DOWNLOAD_URL, ensure_dataset

CLASS_NAMES = ["mandatory", "prohibitory", "warning"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# CCTSDB2021 只有三个大类，保留别名以兼容重标注版本。
NAME_MAP = {
    "mandatory": 0,
    "指示": 0,
    "指示标志": 0,
    "indicator": 0,
    "prohibitory": 1,
    "禁令": 1,
    "禁令标志": 1,
    "prohibition": 1,
    "warning": 2,
    "警告": 2,
    "警告标志": 2,
}


# 生效的类别表：默认 CCTSDB 三类，可由 --classes 覆盖。
ACTIVE_CLASS_NAMES: list[str] = list(CLASS_NAMES)


def configure_classes(names: list[str]) -> None:
    """按 --classes 的顺序重建类别表，类名大小写与连字符不敏感。"""
    cleaned = [name.strip() for name in names if name and name.strip()]
    if not cleaned:
        raise ValueError("--classes 不能为空")
    ACTIVE_CLASS_NAMES[:] = cleaned
    for idx, name in enumerate(cleaned):
        key = name.strip().lower().replace("_", "").replace("-", "")
        NAME_MAP.setdefault(key, idx)


def normalize_class(name: str) -> int | None:
    key = name.strip().lower().replace("_", "").replace("-", "")
    if key in NAME_MAP:
        return NAME_MAP[key]
    # 部分导出用 i1/p2/w3 这类类别编号。
    if len(key) >= 2 and key[0] in "ipw" and key[1:].isdigit():
        return {"i": 0, "p": 1, "w": 2}[key[0]]
    return None


def text_of(node: ET.Element | None, default: str = "") -> str:
    if node is None or node.text is None:
        return default
    return node.text.strip()


def _rotated_box_to_xyxy(node: ET.Element | None) -> list[float] | None:
    """HRSC2016 的 ``<robndbox>``（中心点 + 宽高 + 弧度）转成水平外接矩形。"""
    if node is None:
        return None
    try:
        cx = float(text_of(node.find("cx")))
        cy = float(text_of(node.find("cy")))
        width = float(text_of(node.find("w")))
        height = float(text_of(node.find("h")))
        angle = float(text_of(node.find("angle")))
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    half_w, half_h = width / 2.0, height / 2.0
    xs: list[float] = []
    ys: list[float] = []
    for dx, dy in ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)):
        xs.append(cx + dx * cos_a - dy * sin_a)
        ys.append(cy + dx * sin_a + dy * cos_a)
    return [min(xs), min(ys), max(xs), max(ys)]


def _first_text(node: ET.Element, paths: tuple[str, ...]) -> str:
    """返回第一个非空子节点的文本，兼容不同导出中的大小写/别名标签。"""
    for path in paths:
        child = node.find(path)
        if child is not None and child.text and child.text.strip():
            return child.text.strip()
    return ""


def _is_difficult(node: ET.Element) -> bool:
    return _first_text(node, ("difficult", "Difficult")).lower() in {"1", "true", "yes"}


def _parse_hrsc_objects(root: ET.Element) -> list[dict[str, Any]]:
    """解析 HRSC2016 原生标注：取 <HRSC_Object> 内的 box_* 字段。"""
    objects: list[dict[str, Any]] = []
    for obj in root.iter("HRSC_Object"):
        name = "ship"
        cls = normalize_class(name)
        try:
            xmin = float(_first_text(obj, ("box_xmin", "bBox_xmin")))
            ymin = float(_first_text(obj, ("box_ymin", "bBox_ymin")))
            xmax = float(_first_text(obj, ("box_xmax", "bBox_xmax")))
            ymax = float(_first_text(obj, ("box_ymax", "bBox_ymax")))
        except (TypeError, ValueError):
            continue
        # 退化框在转换前丢弃；越界裁剪交给下游 _build_pair。
        if xmax <= xmin or ymax <= ymin:
            continue
        # 文献口径把 2976 个实例全部计入，故忽略 difficult 标记。
        objects.append(
            {
                "name": name,
                "class": cls,
                "box": [xmin, ymin, xmax, ymax],
                "difficult": False,
            }
        )
    return objects


def parse_xml(xml_path: Path) -> dict[str, Any]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = (
        text_of(root.find("filename"))
        or text_of(root.find("path"))
        or _first_text(root, ("Img_FileName",))
    )
    width = _first_text(root, ("size/width", "Img_SizeWidth"))
    height = _first_text(root, ("size/height", "Img_SizeHeight"))

    objects: list[dict[str, Any]] = []
    for obj in root.findall("object"):
        name = text_of(obj.find("name"))
        cls = normalize_class(name)
        box = obj.find("bndbox")
        if box is not None:
            try:
                xmin = float(text_of(box.find("xmin")))
                ymin = float(text_of(box.find("ymin")))
                xmax = float(text_of(box.find("xmax")))
                ymax = float(text_of(box.find("ymax")))
            except (TypeError, ValueError):
                continue
        else:
            rotated = _rotated_box_to_xyxy(obj.find("robndbox"))
            if rotated is None:
                continue
            xmin, ymin, xmax, ymax = rotated
        if xmax <= xmin or ymax <= ymin:
            continue
        objects.append(
            {
                "name": name,
                "class": cls,
                "box": [xmin, ymin, xmax, ymax],
                "difficult": _is_difficult(obj),
            }
        )

    # HRSC2016 没有 <object> 节点，VOC 分支为空时回退原生解析。
    if not objects:
        objects = _parse_hrsc_objects(root)

    return {
        "filename": filename,
        "width": float(width) if width else None,
        "height": float(height) if height else None,
        "objects": objects,
    }


def build_image_index(raw_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for p in raw_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            index[p.stem].append(p)
    return dict(index)


def choose_image(xml_path: Path, filename: str, index: dict[str, list[Path]]) -> Path | None:
    stem = Path(filename).stem if filename else xml_path.stem
    candidates = index.get(stem, [])
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # 同名图片有多张时，优先取同目录、路径更浅的那张。
    def score(p: Path) -> tuple[int, int]:
        same_dir = 0 if p.parent == xml_path.parent else 1
        return same_dir, len(p.parts)

    return sorted(candidates, key=score)[0]


ANNOTATION_DIRS = {"labels", "label", "labeltxt", "annotations", "annotation", "xmls", "xml"}


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def find_annotations(raw_dir: Path) -> tuple[list[Path], list[Path]]:
    """递归收集 VOC xml 与标注目录（labelTxt/labels）下的 txt。"""
    xml_files = sorted(raw_dir.rglob("*.xml"))
    txt_files = sorted(
        path for path in raw_dir.rglob("*.txt") if path.parent.name.lower() in ANNOTATION_DIRS
    )
    return xml_files, txt_files


def is_polygon_txt(txt_path: Path) -> bool:
    """判断是否是多边形标注：``x1 y1 ... x4 y4 class [difficult]``。"""
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parts = line.split()
                if not parts:
                    continue
                if len(parts) < 9:
                    return False
                return all(_is_number(part) for part in parts[:8]) and not _is_number(parts[8])
    except OSError:
        return False
    return False


def parse_polygon_txt(txt_path: Path) -> list[dict[str, Any]]:
    """把 4 点多边形文本解析成水平外接矩形。"""
    objects: list[dict[str, Any]] = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 9 or not all(_is_number(part) for part in parts[:8]):
                continue
            name = parts[8]
            difficult = len(parts) > 9 and parts[9].strip() == "1"
            if difficult:
                continue
            coords = [float(part) for part in parts[:8]]
            xs = coords[0::2]
            ys = coords[1::2]
            objects.append(
                {
                    "name": name,
                    "class": normalize_class(name),
                    "box": [min(xs), min(ys), max(xs), max(ys)],
                }
            )
    return objects


def _build_pair(
    annotation: Path,
    objects: list[dict[str, Any]],
    stem: str,
    index: dict[str, list[Path]],
    unknown: Counter,
) -> dict[str, Any] | None:
    image_path = choose_image(annotation, stem, index)
    if image_path is None:
        return None

    try:
        from PIL import Image

        with Image.open(image_path) as im:
            width, height = im.size
    except Exception:
        return None
    if not width or not height:
        return None

    boxes: list[list[float]] = []
    labels: list[int] = []
    for obj in objects:
        if obj.get("difficult"):
            continue
        if obj["class"] is None:
            unknown[obj["name"]] += 1
            continue
        x1, y1, x2, y2 = obj["box"]
        x1 = max(0.0, min(float(width - 1), x1))
        y1 = max(0.0, min(float(height - 1), y1))
        x2 = max(0.0, min(float(width - 1), x2))
        y2 = max(0.0, min(float(height - 1), y2))
        if x2 <= x1 + 1.0 or y2 <= y1 + 1.0:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(int(obj["class"]))

    if not boxes:
        return None
    return {
        "annotation": annotation,
        "image": image_path,
        "width": width,
        "height": height,
        "boxes": boxes,
        "labels": labels,
    }


def collect_pairs(raw_dir: Path) -> tuple[list[dict[str, Any]], Counter]:
    root = raw_dir
    index = build_image_index(root)
    xml_files, txt_files = find_annotations(root)
    print(
        f"[prepare] found {len(xml_files)} xml and {len(txt_files)} txt annotations, "
        f"{sum(len(v) for v in index.values())} images"
    )

    pairs: list[dict[str, Any]] = []
    unknown = Counter()
    empty_xml: list[Path] = []
    parse_errors: list[tuple[Path, str]] = []

    for xml_path in xml_files:
        try:
            info = parse_xml(xml_path)
        except (ET.ParseError, OSError) as exc:
            parse_errors.append((xml_path, str(exc)))
            continue
        if not info["objects"]:
            empty_xml.append(xml_path)
            continue
        stem = Path(info["filename"]).stem if info["filename"] else xml_path.stem
        item = _build_pair(xml_path, info["objects"], stem, index, unknown)
        if item is not None:
            pairs.append(item)

    for txt_path in txt_files:
        if not is_polygon_txt(txt_path):
            continue
        item = _build_pair(txt_path, parse_polygon_txt(txt_path), txt_path.stem, index, unknown)
        if item is not None:
            pairs.append(item)

    if empty_xml:
        print(f"[prepare] warning: {len(empty_xml)} xml without boxes (skipped), first 3:")
        for path in empty_xml[:3]:
            print(f"    {path}")
    if parse_errors:
        print(f"[prepare] warning: {len(parse_errors)} xml parse errors (skipped), first 3:")
        for path, message in parse_errors[:3]:
            print(f"    {path}: {message}")
    if unknown:
        print("[prepare] warning: unknown class names (skipped):")
        for name, count in unknown.most_common(20):
            print(f"    {name!r}: {count}")
    return pairs, unknown


def split_tag(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    if {"test", "testing", "eval"} & parts:
        return "test"
    if {"val", "valid", "validation"} & parts:
        return "val"
    if {"train", "training"} & parts:
        return "train"
    return ""


def split_pairs(
    pairs: list[dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    tagged: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for item in pairs:
        tag = split_tag(item["annotation"])
        if tag in tagged:
            tagged[tag].append(item)
        else:
            tagged["train"].append(item)

    # 没有官方划分文件时，按比例随机切出 test 与 val。
    if not tagged["test"]:
        rng.shuffle(tagged["train"])
        n_test = max(1, int(round(len(tagged["train"]) * test_ratio)))
        tagged["test"] = tagged["train"][:n_test]
        tagged["train"] = tagged["train"][n_test:]

    if not tagged["val"]:
        rng.shuffle(tagged["train"])
        n_val = max(1, int(round(len(tagged["train"]) * val_ratio)))
        tagged["val"] = tagged["train"][:n_val]
        tagged["train"] = tagged["train"][n_val:]

    # 防止训练集为空，必要时用 val 回填。
    if not tagged["train"] and tagged["val"]:
        tagged["train"] = tagged["val"]
        tagged["val"] = []

    for key in tagged:
        rng.shuffle(tagged[key])
    return tagged


def unique_stem(stem: str, image_path: Path, used: set[str]) -> str:
    if stem not in used:
        used.add(stem)
        return stem
    digest = hashlib.md5(str(image_path).encode("utf-8")).hexdigest()[:8]
    candidate = f"{stem}_{digest}"
    used.add(candidate)
    return candidate


def export_split(
    items: list[dict[str, Any]],
    split: str,
    out_dir: Path,
    copy_mode: str,
) -> dict[str, Any]:
    image_dir = out_dir / "images" / split
    label_dir = out_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    used: set[str] = set()
    class_counts = [0] * len(ACTIVE_CLASS_NAMES)
    relative_areas: list[float] = []

    for item in items:
        stem = unique_stem(item["image"].stem, item["image"], used)
        suffix = item["image"].suffix.lower()
        dst_image = image_dir / f"{stem}{suffix}"
        dst_label = label_dir / f"{stem}.txt"

        if copy_mode == "hardlink":
            try:
                if dst_image.exists():
                    dst_image.unlink()
                os.link(item["image"], dst_image)
            except OSError:
                shutil.copy2(item["image"], dst_image)
        else:
            shutil.copy2(item["image"], dst_image)

        width, height = item["width"], item["height"]
        lines: list[str] = []
        for (x1, y1, x2, y2), cls in zip(item["boxes"], item["labels"]):
            cx = ((x1 + x2) / 2.0) / width
            cy = ((y1 + y2) / 2.0) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            bw = min(max(bw, 0.0), 1.0)
            bh = min(max(bh, 0.0), 1.0)
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            class_counts[cls] += 1
            relative_areas.append(bw * bh)

        with open(dst_label, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    return {
        "images": len(items),
        "boxes": int(sum(class_counts)),
        "class_counts": class_counts,
        "relative_area": relative_areas,
    }


def make_preview(out_dir: Path, max_images: int = 8) -> None:
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        print(f"[prepare] skip preview (opencv unavailable: {exc})")
        return

    image_dir = out_dir / "images" / "train"
    label_dir = out_dir / "labels" / "train"
    images = sorted(image_dir.glob("*"))[:max_images]
    if not images:
        return

    colors = [(0, 200, 0), (0, 0, 220), (220, 120, 0)]
    thumbs = []
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(image, (x1, y1), (x2, y2), colors[cls % len(colors)], 2)
        target_w = 360
        scale = target_w / max(w, 1)
        image = cv2.resize(image, (target_w, int(h * scale)))
        thumbs.append(image)

    if not thumbs:
        return
    cols = min(4, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    cell_h = max(t.shape[0] for t in thumbs)
    canvas = np.full((rows * cell_h, cols * 360, 3), 255, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        r, c = divmod(idx, cols)
        canvas[r * cell_h : r * cell_h + thumb.shape[0], c * 360 : c * 360 + 360] = thumb
    cv2.imwrite(str(out_dir / "samples.jpg"), canvas)
    print(f"[prepare] wrote {out_dir / 'samples.jpg'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, help="root of the extracted dataset")
    parser.add_argument("--out-dir", default="data/processed/cctsdb")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-mode", choices=["copy", "hardlink"], default="copy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preview", type=int, default=8)
    parser.add_argument(
        "--classes",
        default="",
        help="逗号分隔的类别名，顺序与标签中的类别编号一致；默认 CCTSDB 三类",
    )
    parser.add_argument(
        "--auto-download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="download CCTSDB2021 from Zenodo when --raw-dir has no annotations yet",
    )
    parser.add_argument("--download-dir", default="data/downloads", help="where dataset.rar is cached")
    parser.add_argument("--download-retries", type=int, default=5)
    parser.add_argument("--no-verify-download", dest="verify_download", action="store_false")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if args.classes.strip():
        configure_classes(args.classes.split(","))

    xml_files, txt_files = find_annotations(raw_dir) if raw_dir.is_dir() else ([], [])
    auto_download = args.auto_download and not args.classes.strip()
    if not xml_files and not txt_files:
        if auto_download:
            print(f"[prepare] no annotations under {raw_dir} -> downloading CCTSDB2021 automatically")
            try:
                ensure_dataset(
                    raw_dir,
                    cache_dir=args.download_dir,
                    verify=args.verify_download,
                    retries=args.download_retries,
                )
            except Exception as exc:
                raise SystemExit(
                    f"[prepare] automatic download failed: {exc}\n"
                    f"  download it manually: {DOWNLOAD_URL}\n"
                    f"  extract it to {raw_dir}, then re-run this command"
                )
        else:
            raise SystemExit(
                f"raw dir not found or has no annotations: {raw_dir}\n"
                f"  expected *.xml (VOC) or labelTxt/labels/*.txt (polygon)\n"
                f"  CCTSDB2021 is fetched automatically unless --classes is given"
            )

    if not raw_dir.exists():
        raise SystemExit(f"raw dir not found: {raw_dir}")

    if out_dir.exists() and any((out_dir / "images").glob("*/*")) and not args.overwrite:
        raise SystemExit(
            f"output already exists: {out_dir}\n"
            f"add --overwrite to regenerate, or choose another --out-dir"
        )
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)

    pairs, _ = collect_pairs(raw_dir)
    if not pairs:
        print(
            "[prepare] no valid image/annotation pairs found.\n"
            "  - did the archive extract completely?\n"
            "  - does --raw-dir contain *.xml or labels/labelTxt/*.txt files?\n"
            "  - are image and annotation stems identical?",
            file=sys.stderr,
        )
        raise SystemExit(2)

    splits = split_pairs(pairs, args.val_ratio, args.test_ratio, args.seed)
    print(
        f"[prepare] pairs: train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}"
    )

    stats: dict[str, Any] = {
        "raw_dir": str(raw_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "class_names": list(ACTIVE_CLASS_NAMES),
        "seed": args.seed,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        info = export_split(splits[split], split, out_dir, args.copy_mode)
        areas = info.pop("relative_area")
        if areas:
            import numpy as np

            arr = np.asarray(areas, dtype=np.float32)
            info["relative_area_percentiles"] = {
                "p10": float(np.percentile(arr, 10)),
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
            }
        stats["splits"][split] = info
        print(
            f"[prepare] {split}: {info['images']} images, {info['boxes']} boxes, "
            f"class_counts={info['class_counts']}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "classes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(ACTIVE_CLASS_NAMES) + "\n")

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(ACTIVE_CLASS_NAMES),
        "names": {i: name for i, name in enumerate(ACTIVE_CLASS_NAMES)},
    }
    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    with open(out_dir / "split.json", "w", encoding="utf-8") as f:
        json.dump(
            {k: [item["image"].name for item in v] for k, v in splits.items()},
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    if args.preview > 0:
        make_preview(out_dir, args.preview)

    print(f"[prepare] done: {out_dir}")


if __name__ == "__main__":
    main()
