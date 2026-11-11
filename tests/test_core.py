"""Pruebas de las partes que no requieren modelos pesados."""
from pathlib import Path

import numpy as np
import pytest

from voxlocal.core import audio, markup
from voxlocal.core.voice_library import VoiceLibrary, slugify


# ---------------- markup (texto estructurado)
def test_plain_text_single_segment():
    segs = markup.parse("Hola mundo", "es")
    assert len(segs) == 1 and segs[0].text == "Hola mundo" and segs[0].language == "es"


def test_ssml_break_prosody_lang():
    segs = markup.parse('<speak>Hola <break time="500ms"/> <prosody rate="fast">rápido</prosody> '
                        '<lang xml:lang="en">hello</lang> fin</speak>', "es")
    kinds = [(s.kind, s.text, s.language, s.speed, s.seconds) for s in segs]
    assert kinds[0] == ("speech", "Hola", "es", 1.0, 0.0)
    assert kinds[1][0] == "break" and kinds[1][4] == pytest.approx(0.5)
    assert ("speech", "rápido", "es", 1.15, 0.0) in kinds
    assert ("speech", "hello", "en", 1.0, 0.0) in kinds
    assert kinds[-1][1] == "fin" and kinds[-1][3] == 1.0


def test_short_pause_tag_and_invalid_xml():
    segs = markup.parse("Uno [pausa 1s] dos", "es")
    assert [s.kind for s in segs] == ["speech", "break", "speech"]
    assert segs[1].seconds == 1.0
    bad = markup.parse("<prosody rate='fast'>sin cerrar", "es")
    assert bad[0].kind == "speech" and "sin cerrar" in bad[0].text


def test_split_sentences_respects_limit():
    text = ("Esta es una frase bastante larga, con muchas comas, que excede el límite " * 6).strip()
    parts = markup.split_sentences(text, 100)
    assert all(len(p) <= 101 for p in parts) and "".join(parts).replace(" ", "") == text.replace(" ", "")


# ---------------- audio
def test_audio_roundtrip_and_denoise(tmp_path: Path):
    sr = 22050
    t = np.arange(sr * 2) / sr
    gate = ((t % 0.5) < 0.25).astype(np.float32)          # "habla" intermitente
    sig = (0.5 * np.sin(2 * np.pi * 220 * t) * gate).astype(np.float32)
    noisy = sig + 0.05 * np.random.default_rng(0).standard_normal(sig.size).astype(np.float32)
    p = audio.save_wav(tmp_path / "x.wav", noisy, sr)
    data, sr2 = audio.load_audio(p, 16000)
    assert sr2 == 16000 and abs(audio.duration(data, sr2) - 2.0) < 0.01
    clean = audio.denoise(noisy, sr)
    quiet = gate == 0
    assert np.std(clean[quiet]) < 0.5 * np.std(noisy[quiet])   # el ruido de fondo baja a menos de la mitad


# ---------------- librería de voces
def test_voice_library_crud(tmp_path: Path):
    torch = pytest.importorskip("torch")
    lib = VoiceLibrary(tmp_path / "voices")
    ref = audio.save_wav(tmp_path / "r.wav", np.zeros(22050, dtype=np.float32), 22050)
    gpt, spk = torch.randn(1, 32, 1024), torch.randn(1, 512, 1)
    p = lib.save("Mi Voz Ñandú", gpt, spk, ref, language="es")
    assert p.id == "mi-voz-nandu" and slugify("¡Hola!") == "hola"
    assert (lib.path(p.id) / "embedding.pt").exists() and (lib.path(p.id) / "speaker_embedding.npy").exists()
    p2 = lib.save("Mi Voz Ñandú", gpt, spk, ref)
    assert p2.id == "mi-voz-nandu-2"
    g, s = lib.load_embedding(p.id)
    assert torch.allclose(g, gpt) and torch.allclose(s, spk)
    assert lib.resolve(p.label) == p.id
    lib.rename(p.id, "Otra")
    assert lib.get(p.id).name == "Otra"
    z = lib.export_zip(p.id, tmp_path)
    imp = lib.import_zip(z)
    assert imp.name == "Otra" and imp.id != p.id
    lib.delete(p.id)
    assert {v.id for v in lib.list()} == {p2.id, imp.id}


def test_samples_cache_and_prefs(tmp_path: Path):
    torch = pytest.importorskip("torch")
    lib = VoiceLibrary(tmp_path / "voices")
    src = audio.save_wav(tmp_path / "Mi Muestra.MP3.wav", np.zeros(22050, dtype=np.float32), 22050)
    ref = audio.save_wav(tmp_path / "r.wav", np.zeros(22050, dtype=np.float32), 22050)
    p = lib.save("Voz", torch.randn(1, 32, 1024), torch.randn(1, 512, 1), ref, source_files=[str(src)])
    assert [s.name for s in lib.samples(p.id)] == ["01-mi-muestra-mp3.wav"]
    # caché en memoria: la segunda carga devuelve el mismo objeto
    assert lib.load_embedding(p.id) is lib.load_embedding(p.id)
    # preferencias persistentes entre instancias (simula reiniciar el programa)
    lib.set_prefs(voice=p.id, language="en")
    again = VoiceLibrary(tmp_path / "voices")
    assert again.get_prefs() == {"voice": p.id, "language": "en"}
    assert [v.id for v in again.list()] == [p.id]          # el archivo de preferencias no es una voz
    lib.delete(p.id)
    with pytest.raises(KeyError):
        lib.load_embedding(p.id)


@pytest.mark.parametrize("fmt", ["wav", "mp3", "flac", "ogg"])
def test_export_formats(tmp_path: Path, fmt):
    import soundfile as sf

    sr = 24000
    x = (0.3 * np.sin(np.linspace(0, 2000, sr))).astype(np.float32)
    wav = audio.save_wav(tmp_path / "voz.wav", x, sr)
    out = audio.export(x, sr, tmp_path / "voz.wav", fmt) if fmt == "wav" else audio.convert(wav, fmt)
    assert out.suffix == f".{fmt}" and out.stat().st_size > 0
    data, sr2 = sf.read(str(out))
    assert sr2 == sr and abs(len(data) - sr) < sr * 0.1   # MP3/OGG pueden añadir relleno
