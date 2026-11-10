"""绘制训练曲线、消融柱状图与混淆矩阵。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _moving_average(values: list[float], window: int) -> list[float]:
    """计算居中滑动平均，window<=1 时原样返回。"""
    if window <= 1:
        return list(values)
    half = window // 2
    out = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def plot_curves(log_path: str, output: str, smooth: int = 0, loss_log: bool = False) -> None:
    # 日志里同一 epoch 可能重复出现，这里按 epoch 去重并保留最后一条
    rows: dict[int, dict[str, str]] = {}
    with open(log_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[int(row["epoch"])] = row

    epochs, losses, maps, maps95 = [], [], [], []
    for epoch in sorted(rows):
        row = rows[epoch]
        epochs.append(epoch)
        losses.append(float(row["loss"]))
        maps.append(float(row["val_mAP50"]))
        maps95.append(float(row["val_mAP50_95"]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # loss 单调下降，不做平滑；首轮尖峰改用对数轴呈现
    axes[0].plot(epochs, losses, marker="o", linewidth=1.4)

    if smooth > 1:
        axes[1].plot(epochs, maps, marker="o", linewidth=1, alpha=0.3, label="mAP@0.5 (raw)")
        axes[1].plot(epochs, maps95, marker="s", linewidth=1, alpha=0.3, label="mAP@0.5:0.95 (raw)")
        axes[1].plot(
            epochs,
            _moving_average(maps, smooth),
            linewidth=2.2,
            label=f"mAP@0.5 (window={smooth})",
        )
        axes[1].plot(
            epochs,
            _moving_average(maps95, smooth),
            linewidth=2.2,
            label=f"mAP@0.5:0.95 (window={smooth})",
        )
    else:
        axes[1].plot(epochs, maps, marker="o", label="mAP@0.5")
        axes[1].plot(epochs, maps95, marker="s", label="mAP@0.5:0.95")

    if loss_log:
        axes[0].set_yscale("log")
        axes[0].annotate(
            f"epoch 1: {losses[0]:.2f}",
            xy=(epochs[0], losses[0]),
            xytext=(epochs[0] + 0.6, losses[0] * 0.55),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"),
        )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss" + (" (log)" if loss_log else ""))
    axes[0].set_title("Training loss")
    axes[0].grid(alpha=0.3, which="both")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP")
    axes[1].set_title("Validation mAP")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"saved {output}")


_METRIC_LABELS = {
    "mAP50": "mAP@0.5",
    "mAP50_95": "mAP@0.5:0.95",
    "precision": "P",
    "recall": "R",
    "f1": "F1",
}


def plot_ablation(
    results: list[str],
    labels: list[str] | None,
    output: str,
    metric: str = "mAP50",
    title: str | None = None,
) -> None:
    """绘制消融对比柱状图，用百分数表示并标注纵轴截断。"""
    values = []
    names = []
    for idx, path in enumerate(results):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        values.append(float(data[metric]) * 100.0)
        names.append(labels[idx] if labels and idx < len(labels) else Path(path).stem)

    display = _METRIC_LABELS.get(metric, metric)
    span = max(max(values) - min(values), 1e-6)

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.6), 4.5))
    colors = ["#8ecae6", "#219ebc", "#ffb703", "#fb8500"][: len(names)]
    bars = ax.bar(names, values, color=colors, width=0.62)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + span * 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel(f"{display} (%)")
    ax.set_ylim(min(values) - span * 0.55, max(values) + span * 0.55)
    ax.set_title(title or f"Ablation study ({display})", fontsize=12)
    ax.text(
        0.012,
        0.965,
        "y-axis truncated",
        transform=ax.transAxes,
        fontsize=8,
        style="italic",
        color="#666666",
        va="top",
    )
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=12, ha="right", fontsize=9)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"saved {output}")


def plot_confusion(
    metrics_path: str,
    output: str,
    class_names: list[str] | None = None,
    title: str | None = None,
) -> None:
    """绘制按行归一化的混淆矩阵，格内同时标注计数与占比。"""
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "confusion" not in data:
        raise SystemExit(
            "metrics json has no 'confusion' field; "
            "rerun src.evaluate with --confusion"
        )
    cm = np.asarray(data["confusion"], dtype=np.int64)
    names = (class_names or ["mandatory", "prohibitory", "warning"]) + ["background"]
    row_sum = np.maximum(cm.sum(axis=1, keepdims=True).astype(float), 1.0)
    norm = cm / row_sum

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    caption = "Confusion matrix @ IoU=0.5" + (f" ({title})" if title else "")
    ax.set_title(caption, fontsize=11)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n{norm[i, j] * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if norm[i, j] > 0.55 else "#222222",
            )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("fraction of ground-truth row")
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("curves")
    p1.add_argument("--log", required=True)
    p1.add_argument("--output", default="outputs/figures/train_curves.png")
    p1.add_argument("--smooth", type=int, default=0, help="EMA 平滑跨度（>1 生效，默认 0 不平滑）")
    p1.add_argument("--loss-log", action="store_true", help="loss 轴改用对数刻度")

    p2 = sub.add_parser("ablation")
    p2.add_argument("--results", nargs="+", required=True)
    p2.add_argument("--labels", nargs="+", default=None)
    p2.add_argument("--metric", default="mAP50")
    p2.add_argument("--title", default=None, help="图标题，如「Ablation study — HRSC2016 (mAP@0.5)」")
    p2.add_argument("--output", default="outputs/figures/ablation_map.png")

    p3 = sub.add_parser("confusion")
    p3.add_argument("--metrics", required=True)
    p3.add_argument("--output", default="outputs/figures/confusion.png")
    p3.add_argument("--classes", nargs="+", default=None)
    p3.add_argument("--title", default=None, help="标题括号内的组别，如 CCTSDB2021 或 MDA + Soft-NMS")

    args = parser.parse_args()
    if args.command == "curves":
        plot_curves(args.log, args.output, smooth=args.smooth, loss_log=args.loss_log)
    elif args.command == "ablation":
        plot_ablation(args.results, args.labels, args.output, args.metric, title=args.title)
    elif args.command == "confusion":
        plot_confusion(args.metrics, args.output, args.classes, title=args.title)


if __name__ == "__main__":
    main()
