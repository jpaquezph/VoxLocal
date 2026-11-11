"""Utilidades de audio: carga WAV/MP3/FLAC, limpieza de ruido, recorte y remuestreo."""
from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

SUPPORTED_EXT = {".wav", ".mp3", ".flac"}


def load_audio(path: str | Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Carga un archivo como float32 mono. Remuestrea si se indica target_sr."""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXT:
        raise ValueError(f"Formato no soportado: {path.suffix}. Use WAV, MP3 o FLAC.")
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        data = data.mean(axis=1)
    except Exception:
        # libsndfile antiguo sin MP3: se recurre a librosa (audioread).
        import librosa

        data, sr = librosa.load(str(path), sr=None, mono=True)
        data = data.astype(np.float32)
    if target_sr and sr != target_sr:
        data = resample(data, sr, target_sr)
        sr = target_sr
    return data, sr


def resample(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return data.astype(np.float32)
    from scipy.signal import resample_poly

    g = gcd(sr_in, sr_out)
    return resample_poly(data, sr_out // g, sr_in // g).astype(np.float32)


def denoise(data: np.ndarray, sr: int, strength: float = 0.85) -> np.ndarray:
    """Reducción de ruido ambiente por compuerta espectral (noisereduce)."""
    import noisereduce as nr

    return nr.reduce_noise(y=data, sr=sr, stationary=False, prop_decrease=strength).astype(np.float32)


def trim_silence(data: np.ndarray, sr: int, top_db: float = 35.0) -> np.ndarray:
    import librosa

    trimmed, _ = librosa.effects.trim(data, top_db=top_db)
    return trimmed if trimmed.size > sr * 0.5 else data


def normalize(data: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(data))) if data.size else 0.0
    return data if m < 1e-6 else (data / m * peak).astype(np.float32)


def duration(data: np.ndarray, sr: int) -> float:
    return len(data) / float(sr)


def save_wav(path: str | Path, data: np.ndarray, sr: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(data, -1.0, 1.0), sr, subtype="PCM_16")
    return path


def silence(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(int(max(0.0, seconds) * sr), dtype=np.float32)


# Formatos de descarga (todos se escriben con libsndfile, sin ffmpeg).
EXPORT_FORMATS: dict[str, dict] = {
    "wav": {"label": "WAV (sin pérdida, máxima compatibilidad)", "kw": {"format": "WAV", "subtype": "PCM_16"}},
    "mp3": {"label": "MP3 (liviano, universal)", "kw": {"format": "MP3", "subtype": "MPEG_LAYER_III"}},
    "flac": {"label": "FLAC (sin pérdida, comprimido)", "kw": {"format": "FLAC", "subtype": "PCM_16"}},
    "ogg": {"label": "OGG Vorbis (liviano, web/juegos)", "kw": {"format": "OGG", "subtype": "VORBIS"}},
}


def export(data: np.ndarray, sr: int, path: str | Path, fmt: str) -> Path:
    """Guarda el audio en el formato pedido; la extensión del archivo se ajusta al formato."""
    fmt = fmt.lower()
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"Formato no soportado: {fmt}")
    path = Path(path).with_suffix(f".{fmt}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(data, -1.0, 1.0), sr, **EXPORT_FORMATS[fmt]["kw"])
    return path


def convert(src: str | Path, fmt: str) -> Path:
    """Convierte un archivo ya generado a otro formato (junto al original)."""
    data, sr = sf.read(str(src), dtype="float32", always_2d=False)
    return export(data, sr, src, fmt)
