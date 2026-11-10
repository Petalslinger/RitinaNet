"""构建 CCTSDB 数据集的 Dataset 与 DataLoader。"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

CLASS_NAMES = ["mandatory", "prohibitory", "warning"]


def read_yolo_label(path: Path, width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
    """读取 YOLO 标注并转换成像素坐标的 xyxy 框与类别。"""
    boxes: list[list[float]] = []
    labels: list[int] = []
    if not path.exists():
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
        )

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = (cx - bw / 2.0) * width
            y1 = (cy - bh / 2.0) * height
            x2 = (cx + bw / 2.0) * width
            y2 = (cy + bh / 2.0) * height
            # 归一化坐标需裁剪到图像范围，并丢弃退化框。
            x1 = max(0.0, min(float(width - 1), x1))
            y1 = max(0.0, min(float(height - 1), y1))
            x2 = max(0.0, min(float(width - 1), x2))
            y2 = max(0.0, min(float(height - 1), y2))
            if x2 <= x1 + 1 or y2 <= y1 + 1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(cls)

    if not boxes:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
        )
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)


class CCTSDBDataset(Dataset):
    """按划分读取 CCTSDB 图像及其 YOLO 标签。"""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        train: bool = False,
        hflip_prob: float = 0.5,
        color_jitter: float = 0.15,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split
        self.train = train
        self.hflip_prob = hflip_prob
        self.color_jitter = color_jitter

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"image dir not found: {self.image_dir}\n"
                f"please run: python -m src.prepare_data --raw-dir ... --out-dir {self.root}"
            )

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.images = sorted(
            p for p in self.image_dir.iterdir() if p.suffix.lower() in exts
        )
        if not self.images:
            raise RuntimeError(f"no images in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        boxes, labels = read_yolo_label(
            self.label_dir / f"{image_path.stem}.txt", width, height
        )

        if self.train:
            if random.random() < self.hflip_prob and boxes.numel() > 0:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                # 图像左右翻转后，框的 x 坐标也要镜像。
                x1 = boxes[:, 0].clone()
                x2 = boxes[:, 2].clone()
                boxes[:, 0] = width - x2
                boxes[:, 2] = width - x1
                boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width - 1)
            if self.color_jitter > 0:
                brightness = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
                contrast = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
                image = TF.adjust_brightness(image, brightness)
                image = TF.adjust_contrast(image, contrast)

        image_tensor = TF.pil_to_tensor(image).float().div(255.0)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if boxes.numel()
            else torch.zeros((0,), dtype=torch.float32),
        }
        return image_tensor, target


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, Any]]]):
    """把批次样本按字段转置成元组。"""
    return tuple(zip(*batch))


def build_dataloader(
    root: str | Path,
    split: str,
    batch_size: int = 2,
    workers: int = 4,
    train_aug: bool = False,
    shuffle: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    """构建带批处理与可选数据增强的 DataLoader。"""
    dataset = CCTSDBDataset(root=root, split=split, train=train_aug)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_fn,
        # 仅在 CUDA 可用时启用锁页内存。
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=workers > 0,
    )
