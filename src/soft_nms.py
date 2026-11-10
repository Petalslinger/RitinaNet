"""实现 Soft-NMS，按 IoU 衰减重叠框得分而非直接剔除。"""
from __future__ import annotations

import torch
from torchvision.ops import box_iou


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
    method: str = "gaussian",
    score_threshold: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算单个类别的 Soft-NMS。"""
    if boxes.numel() == 0 or scores.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.long, device=boxes.device),
            torch.empty((0,), dtype=scores.dtype, device=scores.device),
        )

    if boxes.shape[0] != scores.shape[0]:
        raise ValueError(f"boxes/scores mismatch: {boxes.shape[0]} vs {scores.shape[0]}")

    boxes = boxes.detach().float()
    scores = scores.detach().float()

    # 按分数降序处理，高分框优先被保留
    order = scores.argsort(descending=True)
    boxes = boxes[order]
    scores = scores[order]

    keep: list[int] = []
    kept_scores: list[float] = []

    while scores.numel() > 0:
        i = int(torch.argmax(scores).item())
        keep.append(int(order[i].item()))
        kept_scores.append(float(scores[i].item()))

        if scores.numel() == 1:
            break

        ious = box_iou(boxes[i].unsqueeze(0), boxes).squeeze(0)
        if method == "linear":
            decay = torch.where(
                ious > iou_threshold,
                (1.0 - ious).clamp(min=0.0),
                torch.ones_like(ious),
            )
        elif method == "gaussian":
            decay = torch.exp(-(ious * ious) / max(sigma, 1e-6))
        else:
            raise ValueError(f"unknown soft-nms method: {method}")

        scores = scores * decay
        # 当前框已保留，置零避免下一轮重复选中
        scores[i] = 0.0

        mask = scores > score_threshold
        mask[i] = False
        if not bool(mask.any()):
            break
        boxes = boxes[mask]
        scores = scores[mask]
        order = order[mask]

    return (
        torch.tensor(keep, dtype=torch.long, device=boxes.device),
        torch.tensor(kept_scores, dtype=torch.float32, device=boxes.device),
    )
