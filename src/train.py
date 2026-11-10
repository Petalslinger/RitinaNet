"""训练 RetinaNet 与 RetinaNet+MDA，并在验证集上跟踪指标。"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .dataset import CLASS_NAMES, CCTSDBDataset, build_dataloader, collate_fn
from .evaluate import evaluate_model
from .model import build_model
from .utils import (
    AverageMeter,
    capture_rng_state,
    count_parameters,
    ensure_dir,
    get_device,
    load_config,
    restore_rng_state,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数，未给出的项交给配置文件补默认值。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cctsdb_retinanet.yaml")
    parser.add_argument("--data", default=None, help="override data root")
    parser.add_argument("--output", default="outputs/checkpoints/retinanet")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--min-size", type=int, default=None)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--trainable-backbone-layers", type=int, default=None)
    parser.add_argument("--focal-alpha", type=float, default=None)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="force reproducible cuDNN kernels (slower); off by default",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-mda", dest="use_mda", action="store_true", default=None)
    parser.add_argument("--no-mda", dest="use_mda", action="store_false")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--subset-train", type=int, default=0)
    parser.add_argument("--val-max-images", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--clip-grad", type=float, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true", help="run a few iterations for debugging")
    return parser.parse_args()


def _pick(cli_value, cfg: dict, key: str, default):
    if cli_value is not None:
        return cli_value
    return cfg.get(key, default)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    args_dict: dict,
    metrics: dict,
    epoch: int,
    best_map: float,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    rng_state: dict | None = None,
) -> None:
    """保存检查点：best.pth 只存权重，last.pth 额外存优化器与随机数状态以便续训。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_map": best_map,
        "args": args_dict,
        "metrics": metrics,
    }
    if optimizer is not None:
        if not optimizer.state:
            print(
                f"warning: {path.name} saved with empty optimizer state "
                "(no parameter has been updated yet; resuming from it behaves like a fresh start)",
                flush=True,
            )
        payload["optimizer"] = optimizer.state_dict()
        payload["scheduler"] = scheduler.state_dict() if scheduler is not None else None
        if rng_state is not None:
            payload["rng"] = rng_state
    torch.save(payload, path)


def main() -> None:
    """构建数据与模型，完成整个训练和验证流程。"""
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    data_root = args.data or data_cfg.get("root", "data/processed/cctsdb")
    if not (Path(data_root) / "images" / "train").is_dir():
        raise SystemExit(
            f"converted dataset not found: {data_root}\n"
            f"  run: python -m src.prepare_data --raw-dir data/raw/CCTSDB2021 --out-dir {data_root}\n"
            f"  (prepare_data downloads CCTSDB2021 automatically when data/raw is empty)"
        )
    num_classes = int(data_cfg.get("num_classes", len(CLASS_NAMES)))
    class_names = data_cfg.get("class_names", CLASS_NAMES[:num_classes])

    epochs = _pick(args.epochs, train_cfg, "epochs", 12)
    batch_size = _pick(args.batch_size, train_cfg, "batch_size", 2)
    lr = _pick(args.lr, train_cfg, "lr", 0.005)
    momentum = _pick(args.momentum, train_cfg, "momentum", 0.9)
    weight_decay = _pick(args.weight_decay, train_cfg, "weight_decay", 1e-4)
    workers = _pick(args.workers, train_cfg, "workers", 4)
    amp = _pick(args.amp, train_cfg, "amp", True)
    seed = _pick(args.seed, train_cfg, "seed", 42)
    deterministic = _pick(args.deterministic, train_cfg, "deterministic", False)
    val_max_images = _pick(args.val_max_images, train_cfg, "val_max_images", 800)
    val_every = _pick(args.val_every, train_cfg, "val_every", 1)
    clip_grad = _pick(args.clip_grad, train_cfg, "clip_grad", 10.0)
    scheduler_name = str(train_cfg.get("scheduler", "cosine")).lower()
    lr_step = int(train_cfg.get("lr_step", 8))
    lr_gamma = float(train_cfg.get("lr_gamma", 0.1))

    min_size = _pick(args.min_size, model_cfg, "min_size", 512)
    max_size = _pick(args.max_size, model_cfg, "max_size", 640)
    backbone_layers = _pick(
        args.trainable_backbone_layers, model_cfg, "trainable_backbone_layers", 3
    )
    focal_alpha = _pick(args.focal_alpha, model_cfg, "focal_alpha", 0.25)
    focal_gamma = _pick(args.focal_gamma, model_cfg, "focal_gamma", 2.0)
    use_mda = True if args.use_mda is None else args.use_mda
    pretrained = True if args.pretrained is None else args.pretrained
    if args.dry_run and val_max_images and val_max_images > 50:
        val_max_images = 50

    set_seed(int(seed), deterministic=bool(deterministic))
    device = torch.device(args.device) if args.device else get_device()
    output_dir = ensure_dir(args.output)

    print("=" * 70)
    print("RetinaNet dense traffic-sign detection")
    print(f"data       : {data_root}")
    print(f"device     : {device}")
    print(f"use_mda    : {use_mda}")
    print(f"input      : min_size={min_size}, max_size={max_size}")
    print(f"epochs     : {epochs}, batch_size={batch_size}, lr={lr}")
    print(f"amp        : {amp}")
    print(f"scheduler  : {scheduler_name}" + (f" (step {lr_step}, gamma {lr_gamma})" if scheduler_name == "step" else ""))
    print("=" * 70)

    train_loader = build_dataloader(
        data_root,
        split="train",
        batch_size=batch_size,
        workers=workers,
        train_aug=True,
        shuffle=True,
    )
    val_loader = build_dataloader(
        data_root,
        split="val",
        batch_size=batch_size,
        workers=workers,
        train_aug=False,
        shuffle=False,
    )

    if args.subset_train and args.subset_train > 0:
        n = min(args.subset_train, len(train_loader.dataset))
        subset = Subset(train_loader.dataset, list(range(n)))
        train_loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            collate_fn=collate_fn,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        print(f"subset train: {n} images")

    model = build_model(
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        use_mda=bool(use_mda),
        soft_nms_enabled=False,  # Soft-NMS 留到评估阶段再启用
        soft_nms_sigma=0.5,
        soft_nms_iou=0.5,
        score_thresh=0.05,
        pretrained=bool(pretrained),
        trainable_backbone_layers=int(backbone_layers),
        focal_alpha=float(focal_alpha),
        focal_gamma=float(focal_gamma),
    ).to(device)

    print(f"trainable parameters: {count_parameters(model):,}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=float(lr),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
        nesterov=True,
    )
    if scheduler_name == "step":
        # 每 lr_step 个 epoch 把学习率乘 lr_gamma
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(lr_step, 1), gamma=lr_gamma
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(int(epochs), 1), eta_min=float(lr) * 0.01
        )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(amp) and device.type == "cuda")

    start_epoch = 0
    best_map = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "optimizer" not in ckpt:
            raise SystemExit(
                f"{args.resume} only stores model weights (best checkpoints skip optimizer state).\n"
                "  resume from last.pth instead, e.g. --resume outputs/checkpoints/baseline/last.pth"
            )
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_map = float(ckpt.get("best_map", 0.0))
        if "rng" in ckpt:
            restore_rng_state(ckpt["rng"])
        else:
            print(
                "warning: checkpoint stores no RNG state (saved by an older version); "
                "shuffling and augmentation continue from the seeded sequence instead of "
                "the exact one used for the remaining epochs",
                flush=True,
            )
        print(f"resumed from {args.resume}, start_epoch={start_epoch}, best_map={best_map:.4f}")

    args_dict = {
        "data": str(data_root),
        "num_classes": num_classes,
        "class_names": class_names,
        "use_mda": bool(use_mda),
        "min_size": int(min_size),
        "max_size": int(max_size),
        "trainable_backbone_layers": int(backbone_layers),
        "focal_alpha": float(focal_alpha),
        "focal_gamma": float(focal_gamma),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "scheduler": scheduler_name,
        "lr_step": int(lr_step),
        "lr_gamma": float(lr_gamma),
        "seed": int(seed),
        "deterministic": bool(deterministic),
    }
    with open(output_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(args_dict, f, ensure_ascii=False, indent=2)

    log_path = output_dir / "train_log.csv"
    if args.resume is None and log_path.exists():
        # 从头训练先清旧日志，避免曲线混入上一轮记录
        print(f"removing stale log {log_path.name} (fresh run, no --resume)")
        log_path.unlink()
    if not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "lr", "loss", "val_mAP50", "val_mAP50_95", "seconds"])

    for epoch in range(start_epoch, int(epochs)):
        model.train()
        meter = AverageMeter()
        start = time.time()

        for step, (images, targets) in enumerate(train_loader):
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [
                {k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets
            ]
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=bool(amp) and device.type == "cuda"):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            if not bool(torch.isfinite(losses).all()):
                raise RuntimeError(
                    f"non-finite loss at epoch {epoch} step {step}: "
                    f"{ {k: float(v) for k, v in loss_dict.items()} }"
                )

            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
            scaler.step(optimizer)
            scaler.update()

            meter.update(float(losses.item()), len(images))
            if args.dry_run and step >= 4:
                break

            if step % 20 == 0:
                print(
                    f"epoch {epoch + 1}/{epochs} step {step} "
                    f"loss={meter.avg:.4f}",
                    flush=True,
                )

        scheduler.step()
        val_metrics = {"mAP50": 0.0, "mAP50_95": 0.0}
        if (epoch + 1) % max(int(val_every), 1) == 0:
            print(f"validating epoch {epoch + 1} ...", flush=True)
            val_metrics = evaluate_model(
                model,
                val_loader,
                device,
                num_classes=num_classes,
                class_names=class_names,
                max_images=int(val_max_images) if val_max_images else 0,
            )

        elapsed = time.time() - start
        print(
            f"epoch {epoch + 1}/{epochs} done: loss={meter.avg:.4f} "
            f"val_mAP50={val_metrics['mAP50']:.4f} "
            f"val_mAP50_95={val_metrics['mAP50_95']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f} "
            f"time={elapsed:.1f}s",
            flush=True,
        )

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    optimizer.param_groups[0]["lr"],
                    meter.avg,
                    val_metrics["mAP50"],
                    val_metrics["mAP50_95"],
                    elapsed,
                ]
            )

        save_checkpoint(
            output_dir / "last.pth",
            model,
            args_dict,
            val_metrics,
            epoch,
            best_map,
            optimizer=optimizer,
            scheduler=scheduler,
            rng_state=capture_rng_state(),
        )
        if val_metrics["mAP50"] >= best_map:
            best_map = float(val_metrics["mAP50"])
            save_checkpoint(
                output_dir / "best.pth",
                model,
                args_dict,
                val_metrics,
                epoch,
                best_map,
            )
            print(f"new best mAP50={best_map:.4f}, saved best.pth", flush=True)

        if args.dry_run:
            print("dry-run finished")
            break

    print(f"training done. best mAP50={best_map:.4f}")
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()
