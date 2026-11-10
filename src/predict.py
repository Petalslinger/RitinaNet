"""对单张图片或整个目录做推理并画出检测框。"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as TF

from .dataset import CLASS_NAMES
from .model import build_model
from .utils import ensure_dir, get_device

COLORS = [
    (46, 204, 113),
    (231, 76, 60),
    (52, 152, 219),
    (241, 196, 15),
    (155, 89, 182),
]


def load_model(checkpoint: str, soft_nms: bool, score_thresh: float, device: torch.device):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    num_classes = int(args.get("num_classes", 3))
    model = build_model(
        num_classes=num_classes,
        min_size=int(args.get("min_size", 512)),
        max_size=int(args.get("max_size", 640)),
        use_mda=bool(args.get("use_mda", True)),
        soft_nms_enabled=soft_nms,
        soft_nms_sigma=0.5,
        soft_nms_iou=0.5,
        score_thresh=score_thresh,
        pretrained=False,
        trainable_backbone_layers=int(args.get("trainable_backbone_layers", 3)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    names = args.get("class_names", CLASS_NAMES[:num_classes])
    return model, names, num_classes


def draw_detections(image: Image.Image, det: dict, names: list[str]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    boxes = det["boxes"].cpu()
    scores = det["scores"].cpu()
    labels = det["labels"].cpu()

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [float(v) for v in box]
        color = COLORS[int(label) % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{names[int(label)]} {float(score):.2f}"
        try:
            bbox = draw.textbbox((x1, max(0, y1 - 12)), text, font=font)
        except Exception:
            bbox = (x1, max(0, y1 - 12), x1 + 100, max(0, y1 - 12) + 12)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, max(0, y1 - 12)), text, fill=(255, 255, 255), font=font)
    return image


def save_labels(path: Path, det: dict, width: int, height: int) -> None:
    """写 YOLO 格式标签：类别 cx cy w h score（都归一化）。"""
    lines = []
    for box, score, label in zip(det["boxes"], det["scores"], det["labels"]):
        x1, y1, x2, y2 = [float(v) for v in box]
        cx = ((x1 + x2) / 2) / width
        cy = ((y1 + y2) / 2) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        lines.append(f"{int(label)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {float(score):.6f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", required=True, help="image file or directory")
    parser.add_argument("--output", default="outputs/predictions")
    parser.add_argument("--soft-nms", action="store_true")
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="images per forward pass; lower it if CUDA runs out of memory",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    model, names, num_classes = load_model(
        args.checkpoint, args.soft_nms, args.score_thresh, device
    )
    out_dir = ensure_dir(args.output)
    if args.save_txt:
        ensure_dir(out_dir / "labels")

    source = Path(args.source)
    if source.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in exts)
    else:
        files = [source]

    if not files:
        raise SystemExit(f"no images found: {source}")

    total = 0
    with torch.inference_mode():
        for batch in _chunks(files, max(1, args.batch_size)):
            loaded = []
            for image_path in batch:
                image = Image.open(image_path).convert("RGB")
                tensor = TF.pil_to_tensor(image).float().div(255.0).to(device)
                loaded.append((image_path, image, tensor))

            outputs = model([tensor for _, _, tensor in loaded])

            for (image_path, image, _), det in zip(loaded, outputs):
                keep = det["scores"] >= args.score_thresh
                det = {k: v[keep] for k, v in det.items()}
                total += int(keep.sum())

                vis = draw_detections(image.copy(), det, names)
                vis_path = out_dir / f"{image_path.stem}_pred.jpg"
                vis.save(vis_path, quality=95)

                if args.save_txt:
                    save_labels(
                        out_dir / "labels" / f"{image_path.stem}.txt",
                        det,
                        image.width,
                        image.height,
                    )

                print(f"{image_path.name}: {int(keep.sum())} detections -> {vis_path}")

    print(f"done: {len(files)} images, {total} detections, output={out_dir}")


if __name__ == "__main__":
    main()
