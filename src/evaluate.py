"""计算 mAP、P/R/F1、FPS 与混淆矩阵。"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.ops import box_iou

from .dataset import CLASS_NAMES, build_dataloader
from .model import build_model
from .utils import get_device, save_json


def _as_cpu_detection(det: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in det.items()}


def _as_cpu_target(target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in target.items() if k in ("boxes", "labels")}


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    max_images: int = 0,
):
    """逐张前向推理，收集检测结果与对应真值。"""
    model.eval()
    all_detections: list[dict[str, torch.Tensor]] = []
    all_targets: list[dict[str, torch.Tensor]] = []
    seen = 0
    for images, targets in data_loader:
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)
        for det, tgt in zip(outputs, targets):
            all_detections.append(_as_cpu_detection(det))
            all_targets.append(_as_cpu_target(tgt))
        seen += len(images)
        if max_images and seen >= max_images:
            break
    return all_detections, all_targets


def _ap_from_scores(scores: np.ndarray, tp: np.ndarray, num_gt: int) -> tuple[float, np.ndarray, np.ndarray]:
    if num_gt <= 0 or scores.size == 0:
        return 0.0, np.array([0.0, 1.0]), np.array([0.0, 0.0])
    order = np.argsort(-scores, kind="mergesort")
    scores = scores[order]
    tp = tp[order].astype(np.float32)
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / float(num_gt)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # 按 VOC 口径做全点插值，使 PR 曲线单调
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap, recall, precision


def _greedy_match(ious: np.ndarray, iou_threshold: float) -> np.ndarray:
    matched = np.full(ious.shape[0], -1, dtype=np.int64)
    if ious.shape[0] == 0 or ious.shape[1] == 0:
        return matched
    used = np.zeros(ious.shape[1], dtype=bool)
    for i in range(ious.shape[0]):
        best = int(np.argmax(ious[i]))
        if ious[i, best] >= iou_threshold and not used[best]:
            used[best] = True
            matched[i] = best
    return matched


def _build_match_cache(
    detections: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    num_classes: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], np.ndarray]:
    cache: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    num_gt = np.zeros(num_classes, dtype=np.int64)
    for det, tgt in zip(detections, targets):
        det_boxes = det["boxes"].numpy()
        det_scores = det["scores"].numpy()
        det_labels = det["labels"].numpy()
        gt_boxes = tgt["boxes"].numpy()
        gt_labels = tgt["labels"].numpy()
        num_gt += np.bincount(gt_labels, minlength=num_classes)[:num_classes]

        if det_scores.size and gt_boxes.shape[0]:
            ious = box_iou(
                torch.as_tensor(det_boxes, dtype=torch.float32),
                torch.as_tensor(gt_boxes, dtype=torch.float32),
            ).numpy()
        else:
            ious = np.zeros((det_scores.size, gt_boxes.shape[0]), dtype=np.float32)
        cache.append((det_scores, det_labels, gt_labels, ious))
    return cache, num_gt


def _class_matches(
    entry: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    cls: int,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    scores, det_labels, gt_labels, ious = entry
    rows = np.flatnonzero(det_labels == cls)
    if rows.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64)
    rows = rows[np.argsort(-scores[rows], kind="mergesort")]
    cls_scores = scores[rows]
    cols = np.flatnonzero(gt_labels == cls)
    if cols.size == 0:
        return cls_scores, np.full(cls_scores.size, -1, dtype=np.int64)
    return cls_scores, _greedy_match(ious[np.ix_(rows, cols)], iou_threshold)


def compute_metrics(
    detections: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    num_classes: int = 3,
    class_names: list[str] | None = None,
    iou_thresholds: tuple[float, ...] = tuple(np.arange(0.5, 1.0, 0.05)),
    return_pr: bool = False,
    return_confusion: bool = False,
) -> dict[str, Any]:
    """汇总 mAP、P/R/F1、各类指标与可选混淆矩阵。"""
    class_names = class_names or CLASS_NAMES[:num_classes]
    cache, num_gt = _build_match_cache(detections, targets, num_classes)

    aps = []
    pr_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    matched05: list[list[tuple[np.ndarray, np.ndarray]]] | None = None
    for thresh in iou_thresholds:
        value = float(thresh)
        matched = [
            [_class_matches(entry, cls, value) for cls in range(num_classes)]
            for entry in cache
        ]
        ap = np.zeros(num_classes, dtype=np.float32)
        for cls in range(num_classes):
            pairs = [m[cls] for m in matched if m[cls][0].size]
            scores = (
                np.concatenate([p[0] for p in pairs])
                if pairs
                else np.zeros(0, dtype=np.float32)
            )
            tp = (
                np.concatenate([(p[1] >= 0).astype(np.float32) for p in pairs])
                if pairs
                else np.zeros(0, dtype=np.float32)
            )
            ap[cls], rec, prec = _ap_from_scores(scores, tp, int(num_gt[cls]))
            if value == 0.5:
                pr_data[cls] = (rec, prec)
        aps.append(ap)
        if value == 0.5:
            matched05 = matched
    aps = np.stack(aps, axis=0)  # 堆成 [阈值数, 类别数]

    valid = num_gt > 0
    map50 = float(np.mean(aps[0][valid])) if valid.any() else 0.0
    map50_95 = float(np.mean(aps[:, valid])) if valid.any() else 0.0

    if matched05 is None:
        matched05 = [
            [_class_matches(entry, cls, 0.5) for cls in range(num_classes)]
            for entry in cache
        ]
        for cls in range(num_classes):
            pairs = [m[cls] for m in matched05 if m[cls][0].size]
            scores = (
                np.concatenate([p[0] for p in pairs])
                if pairs
                else np.zeros(0, dtype=np.float32)
            )
            tp = (
                np.concatenate([(p[1] >= 0).astype(np.float32) for p in pairs])
                if pairs
                else np.zeros(0, dtype=np.float32)
            )
            _, rec, prec = _ap_from_scores(scores, tp, int(num_gt[cls]))
            pr_data[cls] = (rec, prec)

    all_scores: list[np.ndarray] = []
    all_tp: list[np.ndarray] = []
    for per_class in matched05:
        for cls_scores, cls_matched in per_class:
            if cls_scores.size == 0:
                continue
            all_scores.append(cls_scores)
            all_tp.append((cls_matched >= 0).astype(np.float32))

    best_p = best_r = best_f1 = 0.0
    if all_scores:
        scores_cat = np.concatenate(all_scores)
        order = np.argsort(-scores_cat, kind="mergesort")
        tp = np.concatenate(all_tp)[order]
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(1.0 - tp)
        total_gt = max(int(num_gt.sum()), 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        recall = tp_cum / total_gt
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
        idx = int(np.argmax(f1))
        best_p, best_r, best_f1 = float(precision[idx]), float(recall[idx]), float(f1[idx])

    metrics: dict[str, Any] = {
        "num_images": len(detections),
        "num_gt": int(num_gt.sum()),
        "mAP50": map50,
        "mAP50_95": map50_95,
        "precision": best_p,
        "recall": best_r,
        "f1": best_f1,
        "iou_thresholds": [float(x) for x in iou_thresholds],
        "per_class": {},
    }
    for cls, name in enumerate(class_names):
        class_f1 = 0.0
        class_p = class_r = 0.0
        pair = pr_data.get(cls) if pr_data else None
        if pair is not None:
            rec, prec = pair
            rec = np.asarray(rec, dtype=np.float64)
            prec = np.asarray(prec, dtype=np.float64)
            if rec.size and prec.size:
                f1 = 2.0 * rec * prec / np.maximum(rec + prec, 1e-9)
                idx = int(np.argmax(f1))
                class_f1 = float(f1[idx])
                class_p = float(prec[idx])
                class_r = float(rec[idx])
        metrics["per_class"][name] = {
            "AP50": float(aps[0, cls]),
            "AP75": float(aps[5, cls]) if aps.shape[0] > 5 else float("nan"),
            "AP50_95": float(np.mean(aps[:, cls])),
            "num_gt": int(num_gt[cls]),
            "F1": class_f1,
            "precision": class_p,
            "recall": class_r,
        }

    if return_pr:
        metrics["pr_curves"] = {
            class_names[cls]: {"recall": rec.tolist(), "precision": prec.tolist()}
            for cls, (rec, prec) in pr_data.items()
        }
    if return_confusion:
        metrics["confusion"] = _confusion_matrix(cache, num_classes)
    return metrics


def _confusion_matrix(
    cache: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> list[list[int]]:
    """统计混淆矩阵，行为真实类别、列为预测类别，末行末列为背景。"""
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    for scores, det_labels, gt_labels, ious in cache:
        order = np.argsort(-scores, kind="mergesort")
        if scores.size == 0:
            matched = np.zeros(0, dtype=np.int64)
        elif ious.shape[1]:
            matched = _greedy_match(ious[order], iou_threshold)
        else:
            matched = np.full(scores.size, -1, dtype=np.int64)
        used_gt = np.zeros(len(gt_labels), dtype=bool)
        for i, gi in enumerate(matched):
            label = det_labels[order[i]]
            if gi >= 0:
                cm[gt_labels[gi], label] += 1
                used_gt[gi] = True
            else:
                cm[num_classes, label] += 1
        for gi in range(len(gt_labels)):
            if not used_gt[gi]:
                cm[gt_labels[gi], num_classes] += 1
    return cm.tolist()


def benchmark_fps(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    max_images: int = 100,
    warmup: int = 10,
) -> float:
    """预热后逐张计时，返回每秒可处理图片数。"""
    model.eval()
    images_seen = 0
    times: list[float] = []
    with torch.inference_mode():
        for images, _ in data_loader:
            images = [img.to(device, non_blocking=True) for img in images]
            for img in images:
                if images_seen < warmup:
                    model([img])
                else:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    model([img])
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    times.append(time.perf_counter() - start)
                images_seen += 1
                if images_seen >= warmup + max_images:
                    break
            if images_seen >= warmup + max_images:
                break
    return float(len(times) / max(sum(times), 1e-9))


def evaluate_model(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    num_classes: int = 3,
    class_names: list[str] | None = None,
    max_images: int = 0,
    return_pr: bool = False,
    return_confusion: bool = False,
) -> dict[str, Any]:
    """在给定数据划分上跑完整评估并返回指标。"""
    detections, targets = collect_predictions(model, data_loader, device, max_images=max_images)
    return compute_metrics(
        detections,
        targets,
        num_classes=num_classes,
        class_names=class_names,
        return_pr=return_pr,
        return_confusion=return_confusion,
    )


def _plot_pr(metrics: dict[str, Any], path: str) -> None:
    """绘制各类别的 P-R 曲线并保存为图片。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = metrics.get("pr_curves", {})
    if not curves:
        return
    plt.figure(figsize=(6, 5))
    for name, data in curves.items():
        rec = np.asarray(data["recall"])
        prec = np.asarray(data["precision"])
        ap = metrics["per_class"][name]["AP50"]
        plt.plot(rec, prec, label=f"{name} (AP50={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    """加载检查点并在指定数据划分上评估，输出指标文件。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--soft-nms", action="store_true")
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--output", default="outputs/metrics/eval.json")
    parser.add_argument("--save-pr", default=None)
    parser.add_argument("--fps", action="store_true")
    parser.add_argument("--confusion", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args", {})
    data_root = args.data or train_args.get("data") or "data/processed/cctsdb"
    num_classes = int(train_args.get("num_classes", 3))

    model = build_model(
        num_classes=num_classes,
        min_size=int(train_args.get("min_size", 512)),
        max_size=int(train_args.get("max_size", 640)),
        use_mda=bool(train_args.get("use_mda", True)),
        soft_nms_enabled=args.soft_nms,
        soft_nms_sigma=args.sigma,
        soft_nms_iou=args.iou_thresh,
        score_thresh=args.score_thresh,
        pretrained=False,
        trainable_backbone_layers=int(train_args.get("trainable_backbone_layers", 3)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)

    loader = build_dataloader(
        data_root,
        split=args.split,
        batch_size=args.batch_size,
        workers=args.workers,
        train_aug=False,
        shuffle=False,
    )
    class_names = list(train_args.get("class_names") or CLASS_NAMES[:num_classes])
    metrics = evaluate_model(
        model,
        loader,
        device,
        num_classes=num_classes,
        class_names=class_names[:num_classes],
        max_images=args.max_images,
        return_pr=True,
        return_confusion=args.confusion,
    )

    if args.fps:
        metrics["fps"] = benchmark_fps(model, loader, device, max_images=100)

    metrics["checkpoint"] = str(Path(args.checkpoint).resolve())
    metrics["split"] = args.split
    metrics["soft_nms"] = bool(args.soft_nms)
    metrics["sigma"] = args.sigma
    metrics["score_thresh"] = args.score_thresh

    save_json(metrics, args.output)
    if args.save_pr:
        _plot_pr(metrics, args.save_pr)

    print(f"\ncheckpoint: {args.checkpoint}")
    print(f"split: {args.split}  images: {metrics['num_images']}  soft_nms: {args.soft_nms}")
    print(f"mAP@0.5      : {metrics['mAP50']:.4f}")
    print(f"mAP@0.5:0.95 : {metrics['mAP50_95']:.4f}")
    print(f"precision    : {metrics['precision']:.4f}")
    print(f"recall       : {metrics['recall']:.4f}")
    print(f"F1           : {metrics['f1']:.4f}")
    if args.fps:
        print(f"FPS          : {metrics['fps']:.2f}")
    print("\nper-class:")
    for name, item in metrics["per_class"].items():
        print(f"  {name:12s} AP50={item['AP50']:.4f}  AP75={item['AP75']:.4f}  "
              f"AP50-95={item['AP50_95']:.4f}  F1={item['F1']:.4f}  GT={item['num_gt']}")
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
