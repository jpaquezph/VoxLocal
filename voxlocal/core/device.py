"""Selección de hardware (RNF: CUDA con fallback a CPU cuantizado)."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

from voxlocal.config import SETTINGS

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    device: str            # "cuda" | "cpu"
    name: str
    quantize_cpu: bool     # aplicar cuantización dinámica int8 a modelos torch

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"


@lru_cache(maxsize=1)
def get_device() -> DeviceInfo:
    import torch

    wanted = SETTINGS.device.lower()
    cuda_ok = torch.cuda.is_available()
    if wanted == "cuda" and not cuda_ok:
        log.warning("Se pidió CUDA pero no hay GPU NVIDIA disponible: usando CPU.")
    if cuda_ok and wanted in ("auto", "cuda"):
        # Tensor cores: TF32 en matmul/cudnn.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return DeviceInfo("cuda", torch.cuda.get_device_name(0), False)

    threads = int(os.environ.get("VOXLOCAL_THREADS", "0")) or max(1, (os.cpu_count() or 2) - 1)
    torch.set_num_threads(threads)
    return DeviceInfo("cpu", f"CPU ({threads} hilos)", True)


def quantize_linear_int8(model):
    """Cuantización dinámica int8 de capas Linear (solo CPU)."""
    import torch

    return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
