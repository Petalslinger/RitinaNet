"""定义 RetinaNet 检测模型及其构建入口。"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.ops import clip_boxes_to_image, remove_small_boxes

from .attention import MultiDimAttention
from .soft_nms import soft_nms


class RetinaNetMDA(nn.Module):
    """在 RetinaNet 上接入 MDA 注意力，并可选地替换为 Soft-NMS 后处理。"""

    def __init__(
        self,
        num_classes: int = 3,
        min_size: int = 512,
        max_size: int = 640,
        use_mda: bool = True,
        soft_nms_enabled: bool = False,
        soft_nms_sigma: float = 0.5,
        soft_nms_iou: float = 0.5,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        detections_per_img: int = 300,
        trainable_backbone_layers: int = 3,
        pretrained: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_mda = use_mda
        self.soft_nms_enabled = soft_nms_enabled
        self.soft_nms_sigma = soft_nms_sigma
        self.soft_nms_iou = soft_nms_iou
        self.score_thresh = score_thresh

        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        detector = retinanet_resnet50_fpn_v2(
            weights=weights,
            progress=True,
            num_classes=91,  # 先按 COCO 加载分类头，之后再替换。
            min_size=min_size,
            max_size=max_size,
            trainable_backbone_layers=trainable_backbone_layers,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img,
        )

        cls_head = detector.head.classification_head
        in_channels = cls_head.cls_logits.in_channels
        num_anchors = cls_head.num_anchors
        cls_head.num_classes = num_classes
        cls_head.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=1, stride=1, padding=0
        )
        nn.init.normal_(cls_head.cls_logits.weight, std=0.01)
        nn.init.constant_(cls_head.cls_logits.bias, 0.0)

        if hasattr(cls_head, "loss"):
            cls_head.loss.alpha = focal_alpha
            cls_head.loss.gamma = focal_gamma

        self.detector = detector
        self.attn = nn.ModuleDict()
        if use_mda:
            out_channels = int(detector.backbone.out_channels)
            # RetinaNet 使用 FPN 的 P3..P7 层，对应键 "0".."4"。
            for level in range(5):
                self.attn[str(level)] = MultiDimAttention(out_channels)
            self._hook = detector.backbone.register_forward_hook(self._fpn_hook)

        self._original_postprocess = detector.postprocess_detections
        if soft_nms_enabled:
            # 实例属性会遮蔽类方法，detector.forward 由此调用该绑定方法。
            detector.postprocess_detections = self._soft_postprocess

    def _fpn_hook(self, module, inputs, output):
        """在 FPN 各层输出上依次应用 MDA 注意力。"""
        if not self.use_mda or not isinstance(output, dict):
            return output
        for key in list(output.keys()):
            if key in self.attn:
                output[key] = self.attn[key](output[key])
        return output

    def forward(self, images, targets=None):
        """前向传播，训练时返回损失，推理时返回检测结果。"""
        return self.detector(images, targets)

    @torch.no_grad()
    def _soft_postprocess(self, head_outputs, anchors, image_sizes):
        """用 Soft-NMS 实现 RetinaNet 的 postprocess_detections。"""
        class_logits = head_outputs["cls_logits"]
        box_regression = head_outputs["bbox_regression"]
        num_classes = int(self.detector.head.classification_head.num_classes)
        detections_per_img = int(self.detector.detections_per_img)
        detections = []

        for index in range(len(image_sizes)):
            image_boxes: list[torch.Tensor] = []
            image_scores: list[torch.Tensor] = []
            image_labels: list[torch.Tensor] = []

            for box_reg_level, logits_level, anchors_level in zip(
                [br[index] for br in box_regression],
                [cl[index] for cl in class_logits],
                anchors[index],
            ):
                scores_per_level = torch.sigmoid(logits_level).flatten()
                keep = scores_per_level > self.detector.score_thresh
                if not bool(keep.any()):
                    continue
                scores_per_level = scores_per_level[keep]
                topk_idxs = torch.where(keep)[0]

                num_topk = min(int(self.detector.topk_candidates), int(topk_idxs.numel()))
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                # 每个预测对应一个锚点与一个类别，需还原两者的索引。
                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.detector.box_coder.decode_single(
                    box_reg_level[anchor_idxs], anchors_level[anchor_idxs]
                )
                boxes_per_level = clip_boxes_to_image(boxes_per_level, image_sizes[index])

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            if not image_boxes:
                device = anchors[index][0].device if anchors[index] else torch.device("cpu")
                detections.append(
                    {
                        "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                        "scores": torch.zeros((0,), dtype=torch.float32, device=device),
                        "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                    }
                )
                continue

            boxes = torch.cat(image_boxes, dim=0)
            scores = torch.cat(image_scores, dim=0)
            labels = torch.cat(image_labels, dim=0)

            if boxes.numel() > 0:
                keep_small = remove_small_boxes(boxes, min_size=1e-2)
                boxes = boxes[keep_small]
                scores = scores[keep_small]
                labels = labels[keep_small]

            final_boxes: list[torch.Tensor] = []
            final_scores: list[torch.Tensor] = []
            final_labels: list[torch.Tensor] = []

            # Soft-NMS 按类别独立执行，避免不同类别互相抑制。
            for cls in range(num_classes):
                cls_mask = labels == cls
                if not bool(cls_mask.any()):
                    continue
                cls_boxes = boxes[cls_mask]
                cls_scores = scores[cls_mask]
                keep, kept_scores = soft_nms(
                    cls_boxes,
                    cls_scores,
                    iou_threshold=self.soft_nms_iou,
                    sigma=self.soft_nms_sigma,
                    method="gaussian",
                    score_threshold=max(self.detector.score_thresh, 1e-3),
                )
                if keep.numel() == 0:
                    continue
                final_boxes.append(cls_boxes[keep])
                final_scores.append(kept_scores)
                final_labels.append(
                    torch.full((keep.numel(),), cls, dtype=torch.int64, device=boxes.device)
                )

            if final_boxes:
                boxes_out = torch.cat(final_boxes, dim=0)
                scores_out = torch.cat(final_scores, dim=0)
                labels_out = torch.cat(final_labels, dim=0)
                if scores_out.numel() > detections_per_img:
                    scores_out, topk = scores_out.topk(detections_per_img)
                    boxes_out = boxes_out[topk]
                    labels_out = labels_out[topk]
            else:
                boxes_out = torch.zeros((0, 4), dtype=torch.float32, device=boxes.device)
                scores_out = torch.zeros((0,), dtype=torch.float32, device=boxes.device)
                labels_out = torch.zeros((0,), dtype=torch.int64, device=boxes.device)

            detections.append(
                {"boxes": boxes_out, "scores": scores_out, "labels": labels_out}
            )

        return detections


def build_model(
    num_classes: int = 3,
    min_size: int = 512,
    max_size: int = 640,
    use_mda: bool = True,
    soft_nms_enabled: bool = False,
    soft_nms_sigma: float = 0.5,
    soft_nms_iou: float = 0.5,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.5,
    trainable_backbone_layers: int = 3,
    pretrained: bool = True,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> RetinaNetMDA:
    """按给定超参数构建 RetinaNetMDA 模型。"""
    return RetinaNetMDA(
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        use_mda=use_mda,
        soft_nms_enabled=soft_nms_enabled,
        soft_nms_sigma=soft_nms_sigma,
        soft_nms_iou=soft_nms_iou,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        trainable_backbone_layers=trainable_backbone_layers,
        pretrained=pretrained,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
    )
