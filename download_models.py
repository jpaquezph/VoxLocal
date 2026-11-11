"""Descarga única del modelo XTTS v2 a ./models. Después la app funciona 100 % offline.

    python download_models.py

XTTS v2 se distribuye bajo la Coqui Public Model License (uso no comercial).
"""
from __future__ import annotations

import os
import sys

os.environ["VOXLOCAL_ALLOW_DOWNLOAD"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0"

from voxlocal.config import XTTS_DIR  # noqa: E402


def main() -> int:
    from huggingface_hub import snapshot_download

    print("Al descargar XTTS v2 acepta la Coqui Public Model License (https://coqui.ai/cpml).")
    print("Descargando XTTS v2 (~1.9 GB)…")
    snapshot_download("coqui/XTTS-v2", local_dir=XTTS_DIR,
                      allow_patterns=["config.json", "vocab.json", "model.pth", "speakers_xtts.pth",
                                      "dvae.pth", "mel_stats.pth", "hash.md5"])
    print("\nModelo listo. Ya puede trabajar sin conexión: run.bat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
