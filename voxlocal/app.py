"""Interfaz local (Gradio) de VoxLocal: clonar voz y texto a voz. Solo escucha en 127.0.0.1."""
from __future__ import annotations

import faulthandler
import gc
import logging
import sys
import threading
from datetime import datetime
from pathlib import Path

import voxlocal.config  # noqa: F401  (fuerza modo offline antes que nada)
from voxlocal.config import LANGUAGES, OUTPUT_DIR, XTTS_DIR
from voxlocal.core.audio import EXPORT_FORMATS
from voxlocal.core.voice_library import VoiceLibrary

import gradio as gr

LIB = VoiceLibrary()
log = logging.getLogger("voxlocal")
LOG_FILE = OUTPUT_DIR / "voxlocal.log"
LANG_CHOICES = [(f"{v['name']} ({k})", k) for k, v in LANGUAGES.items()]

SSML_HELP = """**Texto estructurado (opcional):**
`<break time="500ms"/>` pausa · `[pausa 1s]` atajo · `<prosody rate="slow|fast|1.2">…</prosody>` velocidad ·
`<lang xml:lang="en">…</lang>` cambia de idioma dentro del mismo texto · `<p>…</p>` párrafo."""


def voice_choices() -> list[str]:
    return [p.label for p in LIB.list()]


def default_voice() -> str | None:
    """Última voz usada (si sigue existiendo) o la primera guardada."""
    ch = voice_choices()
    last = LIB.get_prefs().get("voice")
    return next((c for c in ch if LIB.resolve(c) == last), ch[0] if ch else None)


def saved_summary() -> str:
    voices = LIB.list()
    if not voices:
        return "Aún no hay voces guardadas. Suba una muestra arriba para crear la primera."
    rows = "\n".join(f"- **{p.name}** · {LANGUAGES.get(p.language, {}).get('name', p.language)} · "
                     f"{p.reference_seconds:.1f}s · creada {p.created[:10]}" for p in voices)
    return (f"**{len(voices)} voz(es) guardada(s)** en la carpeta `voices`. Quedan disponibles cada vez que "
            f"abra el programa: no hace falta volver a subir los audios.\n\n{rows}")


def show_saved(label):
    if not label:
        return None, None
    vid = LIB.resolve(label)
    samples = [str(p) for p in LIB.samples(vid)]
    return str(LIB.path(vid) / "reference.wav"), (samples or None)


def refresh_saved(current=None):
    ch = voice_choices()
    value = current if current in ch else (ch[0] if ch else None)
    return saved_summary(), gr.update(choices=ch, value=value)


def system_status() -> str:
    from voxlocal.core.device import get_device

    dev = get_device()
    model = "✅" if (XTTS_DIR / "model.pth").exists() else "❌ (ejecute download_models.py)"
    return (f"**Dispositivo:** {dev.device.upper()} · {dev.name} · XTTS v2 {model}\n\n"
            "🔒 Ejecución 100 % local: sin APIs de pago ni telemetría.")


# ------------------------------------------------------------------ clonar
def do_clone(files, name, language, denoise, progress=gr.Progress()):
    from voxlocal.core.profiler import create_profile

    if not files:
        raise gr.Error("Cargue al menos un audio WAV, MP3 o FLAC (ideal 6-10 s, una sola persona).")
    if not (name or "").strip():
        raise gr.Error("Escriba un nombre para la voz.")
    progress(0.1, desc="Limpiando ruido y cargando el modelo (la primera vez tarda ~1-2 min)…")
    paths = [f if isinstance(f, str) else f.name for f in files]
    try:
        prof, warnings = create_profile(paths, name, language, denoise)
    except (ValueError, FileNotFoundError) as e:
        raise gr.Error(str(e))
    LIB.set_prefs(voice=prof.id)
    msg = f"✅ Voz **{prof.name}** guardada ({prof.reference_seconds:.1f}s de audio útil). " \
          "Queda guardada para siempre: ya puede usarla en la pestaña *Texto a voz* sin volver a subir audios."
    if warnings:
        msg += "\n\n" + "\n".join(f"⚠ {w}" for w in warnings)
    return msg, str(LIB.path(prof.id) / "reference.wav"), gr.update(choices=voice_choices(), value=prof.label)


def do_delete(label):
    if not label:
        raise gr.Error("Seleccione una voz.")
    LIB.delete(LIB.resolve(label))
    return gr.update(choices=voice_choices(), value=default_voice()), "Voz eliminada."


# ------------------------------------------------------------------ texto a voz
def make_download(wav_path, fmt):
    """Devuelve el archivo de descarga en el formato elegido (convierte si hace falta)."""
    from voxlocal.core import audio

    if not wav_path:
        return None
    fmt = (fmt or "wav").lower()
    src = Path(wav_path)
    if fmt == "wav":
        return str(src)
    target = src.with_suffix(f".{fmt}")
    if not target.exists():
        try:
            target = audio.convert(src, fmt)
        except Exception as e:  # noqa: BLE001
            log.exception("Fallo al convertir a %s", fmt)
            raise gr.Error(f"No se pudo convertir a {fmt.upper()}: {e}")
    return str(target)


def change_format(wav_path, fmt):
    LIB.set_prefs(format=fmt)
    return make_download(wav_path, fmt)


def do_tts(voice, language, text, speed, temperature, fmt, progress=gr.Progress()):
    from voxlocal.core import audio
    from voxlocal.core.tts import XTTSEngine

    if not voice:
        raise gr.Error("Primero clone una voz en la pestaña 1 y selecciónela.")
    if not (text or "").strip():
        raise gr.Error("Escriba el texto a convertir en voz.")
    vid = LIB.resolve(voice)
    log.info("TTS voz=%s idioma=%s chars=%d", vid, language, len(text))
    LIB.set_prefs(voice=vid, language=language)
    try:
        progress(0.1, desc="Cargando el modelo…")
        engine = XTTSEngine.get()
        progress(0.3, desc="Generando voz…")
        wav = engine.synthesize_markup(text, language, LIB.load_embedding(vid), speed, temperature)
        path = audio.save_wav(OUTPUT_DIR / f"tts-{vid}-{language}-{datetime.now():%Y%m%d-%H%M%S-%f}.wav",
                              wav, engine.sr)
    except MemoryError:
        log.exception("Sin memoria durante la síntesis")
        raise gr.Error("No hay memoria RAM suficiente. Cierre otros programas o use un texto más corto.")
    except KeyError:
        raise gr.Error("La voz seleccionada ya no existe. Elija otra.")
    except Exception as e:  # noqa: BLE001
        log.exception("Fallo en la síntesis")
        raise gr.Error(f"No se pudo generar la voz: {type(e).__name__}: {e}. Detalle en outputs/voxlocal.log")
    finally:
        gc.collect()
    log.info("TTS listo: %s", path.name)
    LIB.set_prefs(format=fmt)
    # el reproductor usa el WAV; la descarga sale en el formato elegido
    return str(path), make_download(path, fmt), str(path)


# ------------------------------------------------------------------ UI
def make_theme():
    # Fuentes del sistema: no se piden recursos a Google Fonts (offline).
    return gr.themes.Soft(primary_hue="sky", font=[gr.themes.Font("Segoe UI"), "system-ui", "sans-serif"],
                          font_mono=[gr.themes.Font("Consolas"), "monospace"])


def build() -> gr.Blocks:
    with gr.Blocks(title="VoxLocal · Clonar voz y texto a voz") as ui:
        gr.Markdown("# 🎙️ VoxLocal\nClona una voz y conviértela en texto hablado — **todo en tu equipo, sin internet**.")
        with gr.Accordion("Estado del sistema", open=False):
            status = gr.Markdown()

        with gr.Tab("1 · Clonar voz"):
            gr.Markdown("Suba 6-10 s de una sola persona hablando con claridad (WAV, MP3 o FLAC). "
                        "Solo clone voces propias o con consentimiento expreso de su titular.")
            with gr.Row():
                with gr.Column():
                    c_files = gr.File(label="Audios de muestra", file_count="multiple",
                                      file_types=[".wav", ".mp3", ".flac"])
                    c_name = gr.Textbox(label="Nombre de la voz", placeholder="Ej.: Mi voz")
                    with gr.Row():
                        c_lang = gr.Dropdown(LANG_CHOICES, value="es", label="Idioma de la muestra")
                        c_denoise = gr.Checkbox(value=True, label="Eliminar ruido ambiente")
                    c_btn = gr.Button("Clonar voz", variant="primary")
                with gr.Column():
                    c_msg = gr.Markdown()
                    c_ref = gr.Audio(label="Audio de referencia limpio", type="filepath")

            gr.Markdown("---\n### 💾 Voces guardadas")
            s_summary = gr.Markdown()
            with gr.Row():
                s_voice = gr.Dropdown(choices=voice_choices(), label="Escuchar una voz guardada", interactive=True)
                s_ref = gr.Audio(label="Referencia limpia", type="filepath")
                s_samples = gr.File(label="Audios de muestra originales", file_count="multiple", interactive=False)

        with gr.Tab("2 · Texto a voz"):
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        t_voice = gr.Dropdown(choices=voice_choices(), label="Voz (sus voces guardadas)",
                                              interactive=True, scale=4)
                        t_del = gr.Button("🗑 Borrar voz", size="sm", scale=1)
                    t_text = gr.Textbox(label="Texto", lines=8, placeholder="Escriba cualquier texto…")
                    gr.Markdown(SSML_HELP)
                with gr.Column(scale=2):
                    t_lang = gr.Dropdown(LANG_CHOICES, value="es", label="Idioma del texto")
                    t_speed = gr.Slider(0.6, 1.6, value=1.0, step=0.05, label="Velocidad")
                    t_temp = gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="Expresividad")
                    t_fmt = gr.Radio([(v["label"], k) for k, v in EXPORT_FORMATS.items()], value="wav",
                                     label="Formato de descarga")
                    t_btn = gr.Button("Generar voz", variant="primary")
                    t_audio = gr.Audio(label="Resultado", type="filepath")
                    t_file = gr.File(label="Descargar")
                    gr.Markdown("<small>Puede cambiar el formato después de generar: se convierte al instante, "
                                "sin volver a sintetizar.</small>")
                    t_msg = gr.Markdown()
                    t_last = gr.State()  # ruta del último WAV generado

        c_btn.click(do_clone, [c_files, c_name, c_lang, c_denoise], [c_msg, c_ref, t_voice]) \
            .then(refresh_saved, s_voice, [s_summary, s_voice])
        s_voice.change(show_saved, s_voice, [s_ref, s_samples])
        t_btn.click(do_tts, [t_voice, t_lang, t_text, t_speed, t_temp, t_fmt], [t_audio, t_file, t_last])
        t_fmt.change(change_format, [t_last, t_fmt], t_file)
        t_del.click(do_delete, t_voice, [t_voice, t_msg]).then(refresh_saved, s_voice, [s_summary, s_voice])

        def on_load():
            prefs = LIB.get_prefs()
            lang = prefs.get("language") if prefs.get("language") in LANGUAGES else "es"
            fmt = prefs.get("format") if prefs.get("format") in EXPORT_FORMATS else "wav"
            dv = default_voice()
            return (system_status(), gr.update(choices=voice_choices(), value=dv), lang,
                    saved_summary(), gr.update(choices=voice_choices(), value=dv), fmt)

        ui.load(on_load, None, [status, t_voice, t_lang, s_summary, s_voice, t_fmt])
    return ui


def setup_logging() -> None:
    """Registro en consola y en outputs/voxlocal.log, incluidos cierres nativos (faulthandler)."""
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=[
        logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")])
    global _fault_file
    _fault_file = open(LOG_FILE, "a", encoding="utf-8")  # se mantiene abierto mientras viva el proceso
    faulthandler.enable(_fault_file, all_threads=True)
    sys.excepthook = lambda t, v, tb: log.critical("Excepción no controlada", exc_info=(t, v, tb))
    threading.excepthook = lambda a: log.critical("Excepción en hilo %s", a.thread, exc_info=(
        a.exc_type, a.exc_value, a.exc_traceback))
    log.info("==== VoxLocal iniciado ====")


_fault_file = None


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="VoxLocal (UI local)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    setup_logging()
    build().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=args.port, inbrowser=not args.no_browser, share=False,
        theme=make_theme(), footer_links=[], allowed_paths=[str(OUTPUT_DIR), str(LIB.root)])
    log.info("==== VoxLocal detenido ====")


if __name__ == "__main__":
    main()
