"""RF-01 · Voice Profiling: audio de muestra -> limpieza -> embebido .pt/.npy -> librería."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voxlocal.config import SETTINGS
from voxlocal.core import audio
from voxlocal.core.voice_library import VoiceLibrary, VoiceProfile

REF_SR = 22050  # frecuencia con la que XTTS calcula los latentes


@dataclass
class PreparedReference:
    wav_path: Path
    seconds: float
    warnings: list[str]


def prepare_reference(files: list[str | Path], denoise: bool = True, out_dir: Path | None = None) -> PreparedReference:
    if not files:
        raise ValueError("Cargue al menos un archivo de audio (WAV, MP3 o FLAC).")
    warnings: list[str] = []
    parts: list[np.ndarray] = []
    for f in files:
        data, _ = audio.load_audio(f, REF_SR)
        if denoise:
            data = audio.denoise(data, REF_SR)
        data = audio.normalize(audio.trim_silence(data, REF_SR))
        parts.append(data)
        parts.append(audio.silence(0.25, REF_SR))
    ref = np.concatenate(parts[:-1])
    secs = audio.duration(ref, REF_SR)

    if secs < 3.0:
        raise ValueError(f"El audio útil dura {secs:.1f}s; se requieren al menos 3s (ideal 6-10s).")
    if secs < SETTINGS.ref_min_seconds:
        warnings.append(f"Duración útil {secs:.1f}s: por debajo de los {SETTINGS.ref_min_seconds:.0f}s "
                        "recomendados; la similitud puede ser menor.")
    if secs > SETTINGS.ref_hard_max_seconds:
        ref = ref[: int(SETTINGS.ref_hard_max_seconds * REF_SR)]
        warnings.append(f"Audio recortado a {SETTINGS.ref_hard_max_seconds:.0f}s.")
        secs = SETTINGS.ref_hard_max_seconds
    elif secs > SETTINGS.ref_max_seconds:
        warnings.append(f"Duración {secs:.1f}s: se usarán los fragmentos más representativos "
                        f"(XTTS condiciona con ~{SETTINGS.ref_max_seconds:.0f}s).")

    out_dir = Path(out_dir or tempfile.mkdtemp(prefix="voxref_"))
    wav = audio.save_wav(out_dir / "reference.wav", ref, REF_SR)
    return PreparedReference(wav, secs, warnings)


def create_profile(files: list[str | Path], name: str, language: str = "es", denoise: bool = True,
                   notes: str = "", library: VoiceLibrary | None = None) -> tuple[VoiceProfile, list[str]]:
    from voxlocal.core.tts import XTTSEngine

    library = library or VoiceLibrary()
    prep = prepare_reference(files, denoise)
    gpt, spk = XTTSEngine.get().compute_embedding([prep.wav_path])
    prof = library.save(name, gpt, spk, prep.wav_path, language=language,
                        source_files=[str(f) for f in files], reference_seconds=prep.seconds,
                        denoised=denoise, notes=notes)
    return prof, prep.warnings
