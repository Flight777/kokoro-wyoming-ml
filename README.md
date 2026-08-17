# kokoro-wyoming-ml

[![build-image](https://github.com/Flight777/kokoro-wyoming-ml/actions/workflows/docker.yml/badge.svg)](https://github.com/Flight777/kokoro-wyoming-ml/actions/workflows/docker.yml)
[![GHCR](https://img.shields.io/badge/ghcr.io-kokoro--wyoming--ml-2496ED?logo=docker&logoColor=white)](https://github.com/Flight777/kokoro-wyoming-ml/pkgs/container/kokoro-wyoming-ml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Multilingual GPU [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) TTS
server for Home Assistant, speaking the Wyoming protocol.

One container serves every language Kokoro knows — including **Mandarin done
properly**, with tones.

## Why this exists

Built for a Home Assistant voice stack on Unraid (RTX 5060 Ti, English +
Mandarin household). Every public Kokoro Wyoming wrapper fell down on at least
one of three points:

* **Home Assistant greys the engine out for Chinese.** Existing wrappers
  advertise all 54 voices as `en_US`, so a `zh-CN` pipeline can't select the
  engine at all — the `zf_`/`zm_` voices are in the model but unreachable from HA.
* **Mandarin comes out toneless.** Those wrappers phonemize everything through
  espeak-ng as English. Mandarin is a tonal language and Kokoro's zh voices were
  trained on misaki's tonal IPA, so espeak's output is both wrong and flat.
* **CPU-only, or PyTorch without Blackwell kernels.** Wrappers built on
  PyTorch need a build with `sm_120` support; on a 50-series card the usual
  wheels fall back to CPU or fail outright.

This wrapper fixes all three:

* tags each voice with real language codes (`zh-CN` / `ja` / `fr-FR` / ...), so
  HA offers it for those pipelines
* uses **misaki** — Kokoro's own tonal Mandarin G2P, pure Python — for the
  `zf_`/`zm_` voices, feeding phonemes straight into the model; espeak-ng for
  everything else
* runs on CUDA via onnxruntime-gpu, so there is no PyTorch and no `sm_120`
  problem (Blackwell works out of the box)
* streams audio per sentence, with decimal-safe, CJK-aware sentence splitting
  ("17.3 degrees" stays one utterance; 。！？ split without needing spaces)

It is originally written code (~230 lines), not a fork of an existing wrapper.

## Architecture

A single Wyoming `AsyncServer` answers `Describe` with one `TtsProgram` whose
voices carry per-voice language tags, derived from the Kokoro voice-id prefix
(`a`/`b` → en-US/en-GB, `z` → zh-CN, `j` → ja, `f` → fr-FR, and so on). On
`Synthesize` the text is stripped of markdown noise, split into sentences, and
each sentence is handed to `kokoro_onnx.create_stream`; the float32 audio that
comes back is clipped, scaled to int16 and written out immediately as Wyoming
`AudioChunk` events — so playback starts while later sentences are still
synthesizing. For `z*` voices only, the sentence first goes through misaki's
`ZHG2P`, the result is filtered to Kokoro's `VOCAB`, and it enters the model via
`is_phonemes=True`, bypassing espeak-ng entirely. Everything else phonemizes
through espeak in the voice's own language. A `Semaphore(1)` serializes GPU work
so concurrent requests queue instead of thrashing VRAM.

## Run

Pull the prebuilt image:

    docker run -d --name kokoro-tts --runtime nvidia \
      -p 10210:10210 ghcr.io/Flight777/kokoro-wyoming-ml:latest

or with compose (edit `docker-compose.yml` to use the `image:` line instead of
`build: .`):

    docker compose up -d

or build it yourself:

    docker build -t local/kokoro-wyoming-ml:latest .

The image bundles the model pack (~350 MB). Startup log should show:

    misaki Chinese G2P loaded (tonal)
    active providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
    Advertising languages: cmn, en, en-GB, en-US, ..., zh, zh-CN
    Ready — listening on tcp://0.0.0.0:10210

CPU-only: set `ONNX_PROVIDER=CPUExecutionProvider` and drop the nvidia
runtime (short sentences synthesize in a few hundred ms on a decent CPU).

## Home Assistant

Settings → Devices & services → Add integration → Wyoming Protocol →
host + port 10210. In a pipeline: TTS `kokoro`, pick a voice
(EN: `af_heart`, `am_michael`; ZH: `zf_xiaoxiao`, `zm_yunxi`; also ja/fr/es/hi/it/pt).
Voice is per-pipeline, so one container serves all your languages.

### Screenshots

<!-- Add screenshots of the Wyoming integration and a zh-CN pipeline with a
     Kokoro voice selected. -->

_To be added._

## Env vars

| var | default | notes |
|---|---|---|
| `KOKORO_VOICE` | `zf_xiaoxiao` | fallback when client sends no voice |
| `KOKORO_SPEED` | `1.0` | 0.5–2.0, global |
| `ONNX_PROVIDER` | `CUDAExecutionProvider` | or `CPUExecutionProvider` |
| `URI` | `tcp://0.0.0.0:10210` | |

## Known limitations

* Abbreviations mid-sentence ("e.g.", "Dr.") still cause a sentence split.
* English words embedded in Chinese sentences go through a bridge phonemizer
  and can sound approximate.
* One synthesis at a time (Semaphore(1)); concurrent requests queue.
* `KOKORO_SPEED` is global, not per-voice.

## Credits

Kokoro-82M by hexgrad · ONNX runtime + model export by
[thewh1teagle/kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) ·
Mandarin G2P by [misaki](https://github.com/hexgrad/misaki).
Built for a Home Assistant voice stack that needed one TTS for an
English/Mandarin household.

## License

MIT — see [LICENSE](LICENSE).

Third-party components, all used unmodified:

| Component | License |
|---|---|
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) weights (bundled in the image) | Apache-2.0, by hexgrad |
| [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) | MIT |
| [misaki](https://github.com/hexgrad/misaki) | Apache-2.0 |
| [wyoming](https://github.com/rhasspy/wyoming) | MIT |
| jieba · pypinyin · cn2an · ordered-set · pypinyin-dict | MIT |
| [espeak-ng](https://github.com/espeak-ng/espeak-ng) + [phonemizer-fork](https://pypi.org/project/phonemizer-fork/) | **GPL-3.0-or-later** |

Note on espeak-ng: `kokoro-onnx` depends on `phonemizer-fork` and
`espeakng-loader`, which bring in libespeak-ng — GPL-3.0-or-later. The container
image therefore ships GPL-3.0 components alongside everything else (as does its
Ubuntu base). Nothing in this repository links against or modifies them
directly, and they remain available under their own terms from their upstream
projects. The Mandarin path does not use espeak-ng at all.
