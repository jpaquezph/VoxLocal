# VoxLocal — Clonar voz y texto a voz (offline)

Clona una voz a partir de unos segundos de audio y convierte cualquier texto en voz con ella.
Todo corre **en tu equipo, sin internet y sin APIs de pago** una vez descargado el modelo.

## Instalación (una sola vez, con internet)

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

Crea `.venv` (Python 3.11), instala PyTorch con **CUDA** si hay GPU NVIDIA o **CPU** si no, instala las
dependencias y descarga XTTS v2 (~1.9 GB) con `download_models.py`.

## Uso

```bat
run.bat                      :: abre http://127.0.0.1:7860
```

1. **Clonar voz:** sube 6-10 s de audio (WAV, MP3 o FLAC) de una sola persona, ponle nombre y pulsa *Clonar voz*.
2. **Texto a voz:** elige la voz, escribe el texto, elige el idioma (es, en, pt, fr, de) y pulsa *Generar voz*.
   Los WAV se guardan en `outputs/`.

CLI: `.venv\python.exe -m voxlocal.cli clone|voices|tts --help`.

### Texto estructurado (opcional)

`<break time="500ms"/>` · `[pausa 1s]` · `<prosody rate="slow|fast|1.2">…</prosody>` ·
`<lang xml:lang="en">…</lang>` · `<p>…</p>`

## Estructura

| Archivo | Función |
|---|---|
| `voxlocal/core/profiler.py` | Limpieza de ruido, recorte, validación de duración y creación del perfil |
| `voxlocal/core/tts.py` | Motor XTTS v2: embebido vocal y síntesis |
| `voxlocal/core/markup.py` | Parser de SSML/tags y división en frases |
| `voxlocal/core/voice_library.py` | Voces guardadas en `voices/<id>/` (`embedding.pt`, `.npy`, `reference.wav`) |
| `voxlocal/core/device.py` | CUDA si existe, si no CPU |
| `voxlocal/app.py` | Interfaz Gradio |

## Variables de entorno

| Variable | Por defecto | Uso |
|---|---|---|
| `VOXLOCAL_DEVICE` | `auto` | `cuda` / `cpu` |
| `VOXLOCAL_XTTS_QUANTIZE` | `0` | `1` = XTTS int8 en CPU (experimental) |
| `VOXLOCAL_THREADS` | núcleos − 1 | Hilos de CPU |

## Pruebas

```powershell
.venv\python.exe -m pytest -q
```

## Uso responsable y licencia

- Clone únicamente su propia voz o voces con consentimiento expreso del titular.
- XTTS v2 se distribuye bajo la **Coqui Public Model License** (uso **no comercial**).
