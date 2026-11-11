"""CLI de VoxLocal (offline).

    python -m voxlocal.cli clone --name "Mi voz" muestra.wav [otra.mp3] [--lang es] [--no-denoise]
    python -m voxlocal.cli voices
    python -m voxlocal.cli tts   --voice mi-voz --lang en "Hello world" -o salida.wav
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import voxlocal.config  # noqa: F401


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="voxlocal")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("clone", help="clonar una voz")
    c.add_argument("files", nargs="+"); c.add_argument("--name", required=True)
    c.add_argument("--lang", default="es"); c.add_argument("--no-denoise", action="store_true")

    sub.add_parser("voices", help="listar voces")

    t = sub.add_parser("tts", help="texto a voz")
    t.add_argument("text"); t.add_argument("--voice", required=True); t.add_argument("--lang", default="es")
    t.add_argument("--speed", type=float, default=1.0); t.add_argument("-o", "--out", default=None)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from voxlocal.core.voice_library import VoiceLibrary

    lib = VoiceLibrary()
    if a.cmd == "clone":
        from voxlocal.core.profiler import create_profile

        prof, warns = create_profile(a.files, a.name, a.lang, not a.no_denoise)
        for w in warns:
            print("⚠", w)
        print(f"Voz creada: {prof.id}  ({lib.path(prof.id)})")
    elif a.cmd == "voices":
        for p in lib.list():
            print(f"{p.id:24} {p.name:30} {p.language}  {p.reference_seconds:.1f}s")
    elif a.cmd == "tts":
        from voxlocal.config import OUTPUT_DIR
        from voxlocal.core import audio
        from voxlocal.core.tts import XTTSEngine

        eng = XTTSEngine.get()
        wav = eng.synthesize_markup(a.text, a.lang, lib.load_embedding(lib.resolve(a.voice)), a.speed)
        out = a.out or OUTPUT_DIR / f"tts-{int(time.time())}.wav"
        print("Guardado:", audio.save_wav(out, wav, eng.sr))
    return 0


if __name__ == "__main__":
    sys.exit(main())
