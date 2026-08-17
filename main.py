"""
Wyoming protocol server for Kokoro TTS — multilingual, GPU (onnxruntime CUDA).

Fixes vs. existing wrappers:
  * Advertises each voice with a REAL language tag (zh-CN, ja, fr-FR, ...) so
    Home Assistant offers the engine for non-English pipelines.
  * Uses misaki (Kokoro's own tonal Mandarin G2P) for the zf_/zm_ voices
    instead of espeak-ng, whose Mandarin output is toneless.
  * Sentence splitting is decimal-safe ("17.3" stays one utterance) and
    CJK-punctuation aware.

Env:
  KOKORO_MODEL   default kokoro-v1.0.onnx
  KOKORO_VOICES  default voices-v1.0.bin
  KOKORO_SPEED   default 1.0
  KOKORO_VOICE   default voice when client sends none (default zf_xiaoxiao)
  ONNX_PROVIDER  CUDAExecutionProvider (default here) or CPUExecutionProvider
  URI            default tcp://0.0.0.0:10210
"""

import argparse
import asyncio
import logging
import os
import re
import time
from functools import partial

import numpy as np
from kokoro_onnx import Kokoro
import kokoro_onnx.config as kcfg
from kokoro_onnx.config import VOCAB
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice, TtsVoiceSpeaker
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize

_LOGGER = logging.getLogger("kokoro-wyoming")

# Kokoro voice id prefix -> (Wyoming/HA language tags, espeak-ng phonemizer lang)
LANG_MAP = {
    "a": (["en-US", "en"], "en-us"),
    "b": (["en-GB", "en"], "en-gb"),
    "z": (["zh-CN", "zh", "cmn"], "cmn"),
    "j": (["ja-JP", "ja"], "ja"),
    "f": (["fr-FR", "fr"], "fr-fr"),
    "e": (["es-ES", "es"], "es"),
    "h": (["hi-IN", "hi"], "hi"),
    "i": (["it-IT", "it"], "it"),
    "p": (["pt-BR", "pt"], "pt-br"),
}
DEFAULT_LANG = (["en-US", "en"], "en-us")


def voice_langs(voice_id: str):
    return LANG_MAP.get(voice_id[:1], DEFAULT_LANG)


class ChineseG2P:
    """Kokoro-native Mandarin G2P (misaki, v1.0 IPA-with-tones). espeak-ng's
    Mandarin output is toneless, which makes the zh voices sound flat; misaki
    is what the voices were trained on. Falls back to espeak if unavailable."""

    def __init__(self, en_phonemize):
        self.g2p = None
        try:
            from misaki import zh  # pure python (jieba + pypinyin), no torch
            self.g2p = zh.ZHG2P(version="1.0", en_callable=lambda t: en_phonemize(t, "en-us"))
            _LOGGER.info("misaki Chinese G2P loaded (tonal)")
        except Exception as e:  # pragma: no cover
            _LOGGER.warning("misaki zh unavailable (%s); Chinese will use espeak (toneless)", e)

    def __call__(self, text: str):
        if self.g2p is None:
            return None
        ps, _ = self.g2p(text)
        ps = ps.replace("ʦ", "ts").replace("ʨ", "tɕ").replace("ʣ", "dz")
        dropped = {c for c in ps if c not in VOCAB}
        if dropped:
            _LOGGER.debug("Dropping phonemes not in Kokoro vocab: %r (text=%r)", sorted(dropped), text[:40])
        return "".join(c for c in ps if c in VOCAB)


def clean_text(text: str) -> str:
    """Strip markdown-ish noise LLMs emit; keep CJK punctuation intact."""
    text = re.sub(r"\*\*|__|`+", "", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    return " ".join(text.split()).strip()


# Sentence split:
#   * CJK terminators (。！？；) split with or without following whitespace.
#   * Western terminators (.!?) split ONLY when followed by whitespace, so
#     decimals ("17.3") and times stay inside one utterance.
_SENT_RE = re.compile(r"(?<=[。！？；])\s*|(?<=[.!?])\s+")


def split_sentences(text: str):
    parts = [p.strip() for p in _SENT_RE.split(text) if p and p.strip()]
    return parts or [text]


def build_info(kokoro: Kokoro) -> Info:
    voices = []
    for vid in sorted(kokoro.voices.keys()):
        tags, _ = voice_langs(vid)
        voices.append(
            TtsVoice(
                name=vid,
                description=vid,
                attribution=Attribution(name="hexgrad", url="https://huggingface.co/hexgrad/Kokoro-82M"),
                installed=True,
                version="1.0",
                languages=tags,
                speakers=[TtsVoiceSpeaker(name=vid.split("_", 1)[-1])],
            )
        )
    return Info(
        tts=[
            TtsProgram(
                name="kokoro",
                description="Kokoro-82M multilingual TTS (ONNX, CUDA)",
                attribution=Attribution(name="hexgrad", url="https://huggingface.co/hexgrad/Kokoro-82M"),
                installed=True,
                version="1.0",
                voices=voices,
            )
        ]
    )


class KokoroHandler(AsyncEventHandler):
    def __init__(self, info: Info, kokoro: Kokoro, default_voice: str, speed: float,
                 sem: asyncio.Semaphore, zh_g2p: "ChineseG2P", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.info = info
        self.kokoro = kokoro
        self.default_voice = default_voice
        self.speed = speed
        self.sem = sem
        self.zh_g2p = zh_g2p

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info.event())
            return True
        if Synthesize.is_type(event.type):
            return await self._synth(Synthesize.from_event(event))
        return True

    async def _synth(self, req: Synthesize) -> bool:
        t0 = time.monotonic()
        voice = req.voice.name if (req.voice and req.voice.name) else self.default_voice
        if voice not in self.kokoro.voices:
            _LOGGER.warning("Unknown voice %s, falling back to %s", voice, self.default_voice)
            voice = self.default_voice
        _, espeak_lang = voice_langs(voice)

        text = clean_text(req.text)
        rate = kcfg.SAMPLE_RATE
        await self.write_event(AudioStart(rate=rate, width=2, channels=1).event())
        if not text:
            await self.write_event(AudioStop().event())
            return True

        total = 0
        first = None
        async with self.sem:
            for sentence in split_sentences(text):
                # Chinese: pre-phonemize with misaki (tonal) and hand Kokoro
                # phonemes directly; everything else goes through espeak.
                phon = self.zh_g2p(sentence) if espeak_lang == "cmn" else None
                if phon:
                    stream = self.kokoro.create_stream(phon, voice=voice, speed=self.speed, is_phonemes=True)
                else:
                    stream = self.kokoro.create_stream(sentence, voice=voice, speed=self.speed, lang=espeak_lang)
                async for audio, sr in stream:
                    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                    if first is None:
                        first = time.monotonic()
                    total += len(pcm)
                    await self.write_event(AudioChunk(audio=pcm, rate=sr, width=2, channels=1).event())
        await self.write_event(AudioStop().event())

        t1 = time.monotonic()
        dur = total / (rate * 2)
        _LOGGER.info(
            "voice=%s lang=%s first_audio=%.0fms total=%.0fms audio=%.1fs text=\"%s\"",
            voice, espeak_lang,
            ((first or t1) - t0) * 1000, (t1 - t0) * 1000, dur, text[:60],
        )
        return True


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default=os.getenv("URI", "tcp://0.0.0.0:10210"))
    ap.add_argument("--model", default=os.getenv("KOKORO_MODEL", "kokoro-v1.0.onnx"))
    ap.add_argument("--voices", default=os.getenv("KOKORO_VOICES", "voices-v1.0.bin"))
    ap.add_argument("--voice", default=os.getenv("KOKORO_VOICE", "zf_xiaoxiao"))
    ap.add_argument("--speed", type=float, default=float(os.getenv("KOKORO_SPEED", "1.0")))
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    os.environ.setdefault("ONNX_PROVIDER", "CUDAExecutionProvider")
    _LOGGER.info("Loading Kokoro (%s) provider=%s", args.model, os.environ["ONNX_PROVIDER"])
    kokoro = Kokoro(args.model, args.voices)
    _LOGGER.info("Loaded %d voices; active providers: %s",
                 len(kokoro.voices), kokoro.sess.get_providers())

    zh_g2p = ChineseG2P(kokoro.tokenizer.phonemize)

    # Warm up an English and a Chinese voice so the first real request is fast
    if "af_heart" in kokoro.voices:
        t = time.monotonic()
        kokoro.create("Warm up.", voice="af_heart", speed=1.0, lang="en-us")
        _LOGGER.info("Warmup af_heart/en-us: %.0fms", (time.monotonic() - t) * 1000)
    if "zf_xiaoxiao" in kokoro.voices:
        t = time.monotonic()
        ph = zh_g2p("预热完成。")
        if ph:
            kokoro.create(ph, voice="zf_xiaoxiao", speed=1.0, is_phonemes=True)
        else:
            kokoro.create("预热完成。", voice="zf_xiaoxiao", speed=1.0, lang="cmn")
        _LOGGER.info("Warmup zf_xiaoxiao/cmn: %.0fms", (time.monotonic() - t) * 1000)

    info = build_info(kokoro)
    langs = sorted({l for v in info.tts[0].voices for l in v.languages})
    _LOGGER.info("Advertising languages: %s", ", ".join(langs))

    sem = asyncio.Semaphore(1)
    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready — listening on %s", args.uri)
    await server.run(partial(KokoroHandler, info, kokoro, args.voice, args.speed, sem, zh_g2p))


if __name__ == "__main__":
    asyncio.run(main())
