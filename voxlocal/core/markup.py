"""Parser de texto estructurado (subconjunto SSML + tags cortos) para el Módulo B.

Soportado:
    <speak> ... </speak>                       raíz opcional
    <break time="500ms"/> | <break time="1.5s"/>
    <prosody rate="slow|medium|fast|x-fast|1.2|120%"> ... </prosody>
    <lang xml:lang="en"> ... </lang>  /  <voice language="en"> ... </voice>
    <p>, <s>                                    pausas naturales entre bloques
    [pausa 500ms] / [pause 1s]                  atajo de pausa en texto plano
Si el XML es inválido, se trata todo como texto plano.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

RATE_WORDS = {"x-slow": 0.7, "slow": 0.85, "medium": 1.0, "default": 1.0, "fast": 1.15, "x-fast": 1.3}
SHORT_PAUSE = re.compile(r"\[(?:pausa|pause)\s*([\d.]+)\s*(ms|s)?\]", re.I)
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


@dataclass
class Segment:
    kind: str              # "speech" | "break"
    text: str = ""
    language: str = "es"
    speed: float = 1.0
    seconds: float = 0.0


def parse_time(value: str) -> float:
    m = re.fullmatch(r"\s*([\d.]+)\s*(ms|s)?\s*", value or "")
    if not m:
        return 0.0
    n = float(m.group(1))
    return n / 1000.0 if (m.group(2) or "ms").lower() == "ms" else n


def parse_rate(value: str | None, current: float) -> float:
    if not value:
        return current
    v = value.strip().lower()
    if v in RATE_WORDS:
        return RATE_WORDS[v]
    try:
        r = float(v[:-1]) / 100.0 if v.endswith("%") else float(v)
    except ValueError:
        return current
    return min(2.0, max(0.5, r))


def looks_like_markup(text: str) -> bool:
    return bool(re.search(r"<\s*(speak|break|prosody|lang|voice|p|s)\b", text, re.I) or SHORT_PAUSE.search(text))


def _short_pauses_to_ssml(text: str) -> str:
    return SHORT_PAUSE.sub(lambda m: f'<break time="{m.group(1)}{m.group(2) or "ms"}"/>', text)


def parse(text: str, default_language: str = "es", default_speed: float = 1.0) -> list[Segment]:
    if not looks_like_markup(text):
        return [Segment("speech", text.strip(), default_language, default_speed)] if text.strip() else []

    src = _short_pauses_to_ssml(text.strip())
    if not re.match(r"^\s*<speak\b", src, re.I):
        src = f"<speak>{src}</speak>"
    try:
        root = ET.fromstring(src)
    except ET.ParseError:
        plain = SHORT_PAUSE.sub(" ", re.sub(r"<[^>]+>", " ", text))
        return [Segment("speech", " ".join(plain.split()), default_language, default_speed)]

    out: list[Segment] = []

    def emit(txt: str | None, lang: str, speed: float):
        if txt and txt.strip():
            out.append(Segment("speech", " ".join(txt.split()), lang, speed))

    def walk(el: ET.Element, lang: str, speed: float):
        tag = el.tag.lower().split("}")[-1]
        if tag == "break":
            out.append(Segment("break", seconds=parse_time(el.get("time", "300ms"))))
            emit(el.tail, lang, speed)
            return
        new_lang = el.get(XML_LANG) or el.get("lang") or el.get("language") or lang
        new_speed = parse_rate(el.get("rate"), speed) if tag == "prosody" else speed
        emit(el.text, new_lang, new_speed)
        for child in el:
            walk(child, new_lang, new_speed)
        if tag in ("p", "s"):
            out.append(Segment("break", seconds=0.35 if tag == "p" else 0.15))
        emit(el.tail, lang, speed)

    walk(root, default_language, default_speed)
    # fusiona voz contigua con mismos parámetros
    merged: list[Segment] = []
    for seg in out:
        if (merged and seg.kind == "speech" and merged[-1].kind == "speech"
                and merged[-1].language == seg.language and merged[-1].speed == seg.speed):
            merged[-1].text += " " + seg.text
        else:
            merged.append(seg)
    return merged


def split_sentences(text: str, max_chars: int) -> list[str]:
    """Divide en frases que no superen max_chars (límite de tokens de XTTS)."""
    parts = re.split(r"(?<=[.!?¡¿;:…])\s+", text.strip())
    chunks: list[str] = []
    for p in parts:
        p = p.strip()
        while len(p) > max_chars:
            cut = p.rfind(",", 0, max_chars)
            cut = cut if cut > max_chars // 3 else p.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            chunks.append(p[:cut + 1].strip())
            p = p[cut + 1:].strip()
        if p:
            chunks.append(p)
    return chunks
