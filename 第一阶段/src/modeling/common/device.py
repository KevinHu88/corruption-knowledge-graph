"""PyTorch 设备选择与 MPS 回退工具。"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class DeviceSelectionError(RuntimeError):
    """PyTorch 不可用或指定设备无效。"""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise DeviceSelectionError(
            "缺少 PyTorch；真实模型运行需要安装 torch"
        ) from exc
    return torch


# 中文注释：根据配置和运行环境选择 CPU/CUDA/MPS，并对不可用的显式设备给出清晰错误。
def select_device(requested: str = "auto") -> Any:
    """按 CUDA、MPS、CPU 顺序选择设备。"""

    torch = _torch()
    name = requested.lower()
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            name = "mps"
        else:
            name = "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise DeviceSelectionError("配置要求 CUDA，但当前 CUDA 不可用")
    if name == "mps" and not (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        raise DeviceSelectionError("配置要求 MPS，但当前 MPS 不可用")
    if name not in {"cuda", "mps", "cpu"}:
        raise DeviceSelectionError(f"不支持的设备：{requested}")
    return torch.device(name)


# 中文注释：将模型移动到目标设备；必要时对不支持的加速设备执行受控回退。
def move_model_to_device(
    model: Any,
    requested: str = "auto",
) -> tuple[Any, Any]:
    """移动模型；MPS 算子不兼容时记录日志并回退 CPU。"""

    torch = _torch()
    device = select_device(requested)
    try:
        return model.to(device), device
    except (RuntimeError, NotImplementedError) as exc:
        if device.type != "mps":
            raise DeviceSelectionError(f"模型无法加载到 {device}") from exc
        logger.warning("模型或 CRF 不支持 MPS，已回退 CPU：%s", exc)
        cpu = torch.device("cpu")
        return model.to(cpu), cpu
