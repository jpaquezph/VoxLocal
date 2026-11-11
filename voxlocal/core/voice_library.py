"""RF-05 · Librería de Voces: guardar, nombrar, seleccionar, renombrar y borrar perfiles.

Estructura en disco (una carpeta por perfil)::

    voices/<id>/
        profile.json            metadatos
        embedding.pt            {gpt_cond_latent, speaker_embedding} (torch)
        gpt_cond_latent.npy     mismo embebido en NumPy (portabilidad)
        speaker_embedding.npy
        reference.wav           audio de referencia ya limpio
"""
from __future__ import annotations

import json
import re
import shutil
import unicodedata
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from voxlocal.config import VOICES_DIR

PROFILE_FILE = "profile.json"
PREFS_FILE = "_preferencias.json"
SAMPLES_DIR = "samples"


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s or "voz"


@dataclass
class VoiceProfile:
    id: str
    name: str
    language: str = "es"
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    source_files: list[str] = field(default_factory=list)
    reference_seconds: float = 0.0
    denoised: bool = True
    notes: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} [{self.id}]"


class VoiceLibrary:
    def __init__(self, root: Path = VOICES_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, tuple] = {}  # embebidos ya cargados en memoria

    # ---------- preferencias (última voz/idioma usados) ----------
    def get_prefs(self) -> dict:
        try:
            return json.loads((self.root / PREFS_FILE).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def set_prefs(self, **values) -> None:
        prefs = self.get_prefs() | values
        (self.root / PREFS_FILE).write_text(json.dumps(prefs, ensure_ascii=False, indent=2), encoding="utf-8")

    def samples(self, voice_id: str) -> list[Path]:
        """Audios originales subidos al crear la voz."""
        d = self.path(voice_id) / SAMPLES_DIR
        return sorted(p for p in d.iterdir() if p.is_file()) if d.exists() else []

    # ---------- consulta ----------
    def path(self, voice_id: str) -> Path:
        return self.root / voice_id

    def list(self) -> list[VoiceProfile]:
        out = []
        for d in sorted(self.root.iterdir()):
            f = d / PROFILE_FILE
            if d.is_dir() and f.exists():
                try:
                    out.append(VoiceProfile(**json.loads(f.read_text(encoding="utf-8"))))
                except (TypeError, json.JSONDecodeError):
                    continue
        return sorted(out, key=lambda p: p.name.lower())

    def get(self, voice_id: str) -> VoiceProfile:
        f = self.path(voice_id) / PROFILE_FILE
        if not f.exists():
            raise KeyError(f"Perfil de voz inexistente: {voice_id}")
        return VoiceProfile(**json.loads(f.read_text(encoding="utf-8")))

    def resolve(self, label_or_id: str) -> str:
        """Acepta 'Nombre [id]' (desplegables de la UI) o el id directo."""
        m = re.search(r"\[([^\]]+)\]\s*$", label_or_id or "")
        return m.group(1) if m else (label_or_id or "").strip()

    # ---------- escritura ----------
    def _unique_id(self, name: str) -> str:
        base = slugify(name)
        vid, i = base, 2
        while self.path(vid).exists():
            vid, i = f"{base}-{i}", i + 1
        return vid

    def save(self, name: str, gpt_cond_latent, speaker_embedding, reference_wav: Path,
             language: str = "es", source_files: list[str] | None = None,
             reference_seconds: float = 0.0, denoised: bool = True, notes: str = "") -> VoiceProfile:
        import torch

        name = name.strip() or "Voz sin nombre"
        vid = self._unique_id(name)
        d = self.path(vid)
        d.mkdir(parents=True)
        torch.save({"gpt_cond_latent": gpt_cond_latent.cpu(),
                    "speaker_embedding": speaker_embedding.cpu()}, d / "embedding.pt")
        np.save(d / "gpt_cond_latent.npy", gpt_cond_latent.cpu().numpy())
        np.save(d / "speaker_embedding.npy", speaker_embedding.cpu().numpy())
        shutil.copy2(reference_wav, d / "reference.wav")
        # conserva los audios de muestra originales
        sd = d / SAMPLES_DIR
        sd.mkdir()
        for i, src in enumerate(source_files or [], 1):
            src = Path(src)
            if src.is_file():
                shutil.copy2(src, sd / f"{i:02d}-{slugify(src.stem)}{src.suffix.lower()}")
        prof = VoiceProfile(id=vid, name=name, language=language,
                            source_files=[Path(s).name for s in (source_files or [])],
                            reference_seconds=round(reference_seconds, 2),
                            denoised=denoised, notes=notes)
        self._write(prof)
        return prof

    def _write(self, prof: VoiceProfile) -> None:
        (self.path(prof.id) / PROFILE_FILE).write_text(
            json.dumps(asdict(prof), ensure_ascii=False, indent=2), encoding="utf-8")

    def rename(self, voice_id: str, new_name: str) -> VoiceProfile:
        prof = self.get(voice_id)
        prof.name = new_name.strip() or prof.name
        self._write(prof)
        return prof

    def delete(self, voice_id: str) -> None:
        d = self.path(voice_id)
        if not (d / PROFILE_FILE).exists():
            raise KeyError(voice_id)
        shutil.rmtree(d)
        self._cache.pop(voice_id, None)

    def load_embedding(self, voice_id: str):
        """Devuelve (gpt_cond_latent, speaker_embedding) como tensores torch (con caché en memoria)."""
        if voice_id in self._cache:
            return self._cache[voice_id]
        import torch

        d = self.path(voice_id)
        if not (d / PROFILE_FILE).exists():
            raise KeyError(voice_id)
        pt = d / "embedding.pt"
        if pt.exists():
            data = torch.load(pt, map_location="cpu", weights_only=True)
            emb = data["gpt_cond_latent"], data["speaker_embedding"]
        else:
            emb = (torch.from_numpy(np.load(d / "gpt_cond_latent.npy")),
                   torch.from_numpy(np.load(d / "speaker_embedding.npy")))
        self._cache[voice_id] = emb
        return emb

    # ---------- portabilidad ----------
    def export_zip(self, voice_id: str, dest_dir: Path) -> Path:
        d = self.path(voice_id)
        dest = Path(dest_dir) / f"voz-{voice_id}.zip"
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
            for f in d.iterdir():
                z.write(f, f.name)
        return dest

    def import_zip(self, zip_path: Path) -> VoiceProfile:
        with zipfile.ZipFile(zip_path) as z:
            names = set(z.namelist())
            if PROFILE_FILE not in names or not ({"embedding.pt", "speaker_embedding.npy"} & names):
                raise ValueError("El ZIP no contiene un perfil de voz válido.")
            meta = json.loads(z.read(PROFILE_FILE).decode("utf-8"))
            vid = self._unique_id(meta.get("name", "importada"))
            d = self.path(vid)
            d.mkdir(parents=True)
            for n in names:
                if "/" in n or "\\" in n or n.startswith("."):
                    continue  # evita path traversal
                (d / n).write_bytes(z.read(n))
        meta["id"] = vid
        prof = VoiceProfile(**{k: v for k, v in meta.items() if k in VoiceProfile.__dataclass_fields__})
        self._write(prof)
        return prof
