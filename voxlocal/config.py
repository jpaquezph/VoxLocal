"""Configuración central de VoxLocal.

Todo se ejecuta en local. Antes de importar cualquier librería de HuggingFace /
Gradio se fuerzan los modos offline y se desactiva la telemetría (RNF: cero
costo operativo, sin llamadas a servicios en la nube).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("VOXLOCAL_MODELS_DIR", ROOT_DIR / "models"))
VOICES_DIR = Path(os.environ.get("VOXLOCAL_VOICES_DIR", ROOT_DIR / "voices"))
OUTPUT_DIR = Path(os.environ.get("VOXLOCAL_OUTPUT_DIR", ROOT_DIR / "outputs"))

XTTS_DIR = MODELS_DIR / "xtts_v2"


def enforce_offline() -> None:
    """Bloquea descargas y telemetría. Se omite durante download_models.py."""
    if os.environ.get("VOXLOCAL_ALLOW_DOWNLOAD") == "1":
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
# Licencia CPML de XTTS v2 (uso no comercial): se acepta explícitamente al
# descargar el modelo; esto evita el prompt interactivo de Coqui.
os.environ.setdefault("COQUI_TOS_AGREED", "1")
enforce_offline()


# Idiomas soportados (códigos de XTTS).
LANGUAGES: dict[str, dict[str, str]] = {
    "es": {"name": "Español"},
    "en": {"name": "Inglés"},
    "pt": {"name": "Portugués"},
    "fr": {"name": "Francés"},
    "de": {"name": "Alemán"},
}


@dataclass
class Settings:
    # "auto" -> CUDA si existe, si no CPU. También "cuda" o "cpu".
    device: str = os.environ.get("VOXLOCAL_DEVICE", "auto")
    # Cuantización dinámica int8 de las capas lineales de XTTS en CPU (experimental).
    xtts_cpu_quantize: bool = os.environ.get("VOXLOCAL_XTTS_QUANTIZE", "0") == "1"
    tts_sample_rate: int = 24000
    # Audio de referencia para el perfil vocal (RF-01).
    ref_min_seconds: float = 6.0
    ref_max_seconds: float = 10.0
    ref_hard_max_seconds: float = 30.0
    languages: dict = field(default_factory=lambda: LANGUAGES)


SETTINGS = Settings()

for _d in (MODELS_DIR, VOICES_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
