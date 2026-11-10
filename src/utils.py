"""训练与评估脚本共用的通用小工具。"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    """读取 YAML 配置文件并返回字典。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | os.PathLike) -> Path:
    """创建目录（含缺失的父目录）并返回 Path 对象。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int, deterministic: bool = False) -> None:
    """固定 Python、NumPy 与 PyTorch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # 完全可复现：固定卷积算法并关闭 benchmark
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # 完全确定性会明显拖慢训练，这里只固定主要随机源
        torch.backends.cudnn.benchmark = True


def capture_rng_state() -> dict:
    """备份所有影响训练的随机数状态，供断点续训精确还原。"""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    """恢复此前备份的随机数状态。"""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def get_device() -> torch.device:
    """按设备可用性自动选择 cuda 或 cpu。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj: Any, path: str | os.PathLike) -> None:
    """写出 JSON 文件，保留中文并按两空格缩进。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | os.PathLike) -> Any:
    """从 JSON 文件读取对象。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module) -> int:
    """统计模型中需要梯度的参数总量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """累计单次观测值并维护总和、计数与均值。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """清空全部统计量。"""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """累加一次观测（默认权重 1）并刷新均值。"""
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)
