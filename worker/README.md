# whisper-hub engine worker

Python service that runs one speech recognition engine behind the internal
Worker API (`docs/DESIGN.md`, section "Worker API"): `GET /healthz`,
`GET /v1/info`, `POST /v1/transcribe` (NDJSON stream), `POST /v1/detect-language`.
It receives 16 kHz mono s16le PCM from the gateway and needs no ffmpeg.

Engines: `faster_whisper` (no torch), `whisperx`, `openai_whisper`, and `fake`
(no model, for gateway tests). When `legacy_output` is set, `done.rendered` is
byte-identical to what upstream whisper-asr-webservice returns from `/asr`
for the same engine, because the same writer code and engine calls are used.

## Build

One engine per image, selected with `ENGINE`:

```shell
docker build --build-arg ENGINE=faster_whisper -t whisper-hub-worker:faster_whisper worker
docker build --build-arg ENGINE=fake           -t whisper-hub-worker:fake worker
docker build --build-arg ENGINE=whisperx       -t whisper-hub-worker:whisperx worker
docker build --build-arg ENGINE=openai_whisper -t whisper-hub-worker:openai_whisper worker
# CUDA:
docker build -f worker/Dockerfile.gpu --build-arg ENGINE=faster_whisper -t whisper-hub-worker:faster_whisper-gpu worker
```

Library versions are pinned to the reference image through
`requirements/constraints.txt` (its `pip freeze`).

## Run

```shell
docker run --rm -p 9100:9100 -e ASR_MODEL=tiny -v whisper-models:/root/.cache/whisper \
  whisper-hub-worker:faster_whisper
docker run --rm -p 9100:9100 whisper-hub-worker:fake
```

## Configuration

Upstream variables keep upstream semantics:

| Variable | Default | Notes |
|---|---|---|
| `ASR_ENGINE` | set by the image | `faster_whisper`, `whisperx`, `openai_whisper`, `fake` |
| `ASR_MODEL` | `base` | name, local path, HF repo id, or HF URL (faster_whisper, whisperx) |
| `ASR_MODEL_PATH` | `~/.cache/whisper` | download root (whisperx ignores it, as upstream) |
| `ASR_DEVICE` | `cuda` if available else `cpu` | openai_whisper uses CUDA when available, as upstream |
| `ASR_QUANTIZATION` | `float32` with CUDA else `int8` | any CTranslate2 compute type |
| `MODEL_IDLE_TIMEOUT` | `0` | seconds idle before a model is unloaded; 0 = never |
| `HF_TOKEN` | empty | Hub downloads; enables whisperx diarization |
| `SUBTITLE_MAX_LINE_WIDTH` / `_COUNT` / `SUBTITLE_HIGHLIGHT_WORDS` | `1000` / `2` / `false` | whisperx srt/vtt only |
| `LOG_LEVEL` | `INFO` | `DEBUG` logs every request, queueing step and segment |

Worker settings:

| Variable | Default | Notes |
|---|---|---|
| `WORKER_HOST`, `WORKER_PORT` | `0.0.0.0`, `9100` | listen address |
| `WORKER_MODELS` | `ASR_MODEL` | comma-separated models this worker serves; the first is the default |
| `WORKER_MAX_LOADED_MODELS` | `1` | loaded at once; the least recently used idle model is evicted |
| `WORKER_PRELOAD` | `false` | load the default model at startup instead of on first use |

Models load lazily. Each model runs one job at a time; further jobs queue.
A client disconnect stops the job: between segments for faster_whisper,
between 30 s windows for openai_whisper, between transcribe, align and
diarize stages for whisperx.

## Fake engine

Deterministic output, no model. Segment `i` (0-based) has `id` i+1, `seek`
i*200, `start` 2i, `end` 2i+2, tokens `[50364+i]`, `avg_logprob` -0.25,
`compression_ratio` 1.2, `no_speech_prob` 0.01, `temperature` 0.0, text cycling:

1. `" Hello from the fake engine."`
2. `" This is segment number two."`
3. `" The quick brown fox jumps over the lazy dog."`

- Language: `options.language` with probability 1.0, else `WORKER_FAKE_LANGUAGE`
  (default `en`) with 0.99. `task` is ignored. `/v1/detect-language` returns
  `WORKER_FAKE_LANGUAGE` with confidence 0.99.
- `word_timestamps`: words split on spaces (with the leading space, as Whisper),
  spread evenly over the segment, probability 0.9; otherwise `words` is null.
- `legacy_output`: rendered exactly as upstream faster_whisper would render
  these segments.
- `WORKER_FAKE_SEGMENTS` (default 3) segments, `WORKER_FAKE_DELAY` seconds
  (default 0) waited before each one; the wait ends at once on disconnect.
- A `prompt` starting with `fake:` carries directives (comma or space separated):
  `delay=<s>`, `segments=<n>`, `fail=bad_request|model_not_found|unavailable|internal`
  (HTTP error before any output), `fail_after=<n>` (n segments, then an
  `error` line with code `internal`).
  Example: `fake:delay=0.5,segments=20`.

## Test

```shell
docker build --build-arg ENGINE=fake --target test -t whisper-hub-worker:fake-test worker
docker run --rm whisper-hub-worker:fake-test
```

The `faster_whisper` test image additionally checks the copied writer code
against the installed library. Legacy fidelity against a running upstream
container (same raw PCM sent to both, upstream with `encode=false`):

```shell
python3 worker/scripts/fidelity.py --ref http://127.0.0.1:9000 --worker http://127.0.0.1:9100 a.pcm b.pcm
```
