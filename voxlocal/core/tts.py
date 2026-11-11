"""Clonación de voz y texto a voz con Coqui XTTS v2 (zero-shot, local).

- Extracción del embebido vocal (gpt_cond_latent + speaker_embedding).
- Síntesis multi-idioma con la voz clonada (es, en, pt, fr, de).
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from voxlocal.config import SETTINGS, XTTS_DIR
from voxlocal.core import audio, markup
from voxlocal.core.device import get_device, quantize_linear_int8

log = logging.getLogger(__name__)

# Límites de caracteres por frase que acepta XTTS v2 (con margen).
CHAR_LIMITS = {"es": 220, "en": 230, "pt": 190, "fr": 240, "de": 230}


class XTTSEngine:
    _instance: "XTTSEngine | None" = None
    _cls_lock = threading.Lock()

    def __init__(self, model_dir: Path = XTTS_DIR):
        self.model_dir = Path(model_dir)
        self.model = None
        self.dev = get_device()
        self.sr = SETTINGS.tts_sample_rate
        self._lock = threading.Lock()  # el modelo no es thread-safe

    @classmethod
    def get(cls) -> "XTTSEngine":
        with cls._cls_lock:
            if cls._instance is None:
                cls._instance = cls()
            cls._instance.load()
            return cls._instance

    # ------------------------------------------------------------------
    def load(self) -> None:
        if self.model is not None:
            return
        cfg_path = self.model_dir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"No se encontró XTTS v2 en {self.model_dir}. Ejecute una vez: python download_models.py")
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        t0 = time.time()
        cfg = XttsConfig()
        cfg.load_json(str(cfg_path))
        model = Xtts.init_from_config(cfg)
        model.load_checkpoint(cfg, checkpoint_dir=str(self.model_dir), use_deepspeed=False, eval=True)
        if self.dev.is_cuda:
            model.cuda()
        elif SETTINGS.xtts_cpu_quantize:
            try:
                model.gpt = quantize_linear_int8(model.gpt)
                log.info("XTTS GPT cuantizado a int8 (CPU).")
            except Exception as e:  # noqa: BLE001
                log.warning("No se pudo cuantizar XTTS: %s", e)
        self.sr = int(cfg.audio.output_sample_rate)
        self.model = model
        log.info("XTTS v2 cargado en %s en %.1fs", self.dev.device, time.time() - t0)

    # ------------------------------------------------------------------
    # RF-01 · Embebido vocal
    def compute_embedding(self, wav_paths: list[str | Path]):
        """Devuelve (gpt_cond_latent, speaker_embedding) en CPU."""
        with self._lock:
            gpt, spk = self.model.get_conditioning_latents(
                audio_path=[str(p) for p in wav_paths],
                gpt_cond_len=int(SETTINGS.ref_max_seconds),
                max_ref_length=int(SETTINGS.ref_hard_max_seconds),
                sound_norm_refs=False,
            )
        return gpt.cpu(), spk.cpu()

    def _to_device(self, latents):
        gpt, spk = latents
        dev = "cuda" if self.dev.is_cuda else "cpu"
        return gpt.to(dev), spk.to(dev)

    # ------------------------------------------------------------------
    # RF-02 · Síntesis
    def synthesize(self, text: str, language: str, latents, speed: float = 1.0,
                   temperature: float = 0.7) -> np.ndarray:
        """Texto plano -> audio float32 mono a self.sr."""
        gpt, spk = self._to_device(latents)
        pieces: list[np.ndarray] = []
        for sent in markup.split_sentences(text, CHAR_LIMITS.get(language, 220)):
            with self._lock:
                out = self.model.inference(sent, language, gpt, spk, temperature=temperature,
                                           speed=float(speed), enable_text_splitting=False)
            pieces.append(np.asarray(out["wav"], dtype=np.float32))
            pieces.append(audio.silence(0.12, self.sr))
        return np.concatenate(pieces) if pieces else audio.silence(0.1, self.sr)

    def synthesize_markup(self, text: str, default_language: str, latents, speed: float = 1.0,
                          temperature: float = 0.7) -> np.ndarray:
        """Texto plano o estructurado (SSML/tags) -> audio."""
        segs = markup.parse(text, default_language, speed)
        chunks = []
        for seg in segs:
            if seg.kind == "break":
                chunks.append(audio.silence(seg.seconds, self.sr))
            else:
                lang = seg.language.split("-")[0].lower()
                if lang not in SETTINGS.languages:
                    lang = default_language
                chunks.append(self.synthesize(seg.text, lang, latents, seg.speed, temperature))
        return np.concatenate(chunks) if chunks else audio.silence(0.1, self.sr)

    def stream(self, text: str, language: str, latents, speed: float = 1.0,
               chunk_size: int = 20) -> Iterator[np.ndarray]:
        """Síntesis incremental: entrega el primer fragmento lo antes posible."""
        gpt, spk = self._to_device(latents)
        for sent in markup.split_sentences(text, CHAR_LIMITS.get(language, 220)):
            with self._lock:
                for chunk in self.model.inference_stream(sent, language, gpt, spk,
                                                         stream_chunk_size=chunk_size,
                                                         speed=float(speed),
                                                         enable_text_splitting=False):
                    yield chunk.detach().cpu().numpy().astype(np.float32).reshape(-1)
