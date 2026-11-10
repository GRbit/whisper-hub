# whisper-hub engine worker

The worker is the Python half of [whisper-hub](../README.md). It runs one
speech recognition engine and serves one or more models to the gateway over
an internal HTTP API, specified in
[docs/worker-api.md](../docs/worker-api.md): `GET /healthz`, `GET /v1/info`,
`POST /v1/transcribe` (a stream of JSON lines) and
`POST /v1/detect-language`.

The gateway decodes all audio. The worker receives 16 kHz mono signed 16-bit
PCM and needs no ffmpeg. Clients never talk to the worker: it has no
authentication, so keep its port on a private network.

You only need this document for the split setup, where gateway and workers
run in separate containers. The all-in-one `whisper-hub` images contain a
worker already; there, the variables below are set on that container.

## Engines

| `WHISPER_HUB_ENGINE` | Library | Notes |
|---|---|---|
| `faster_whisper` | faster-whisper (CTranslate2) | no torch, smallest image; streams segments as they are decoded |
| `openai_whisper` | openai-whisper (torch) | streams segments after each 30-second window |
| `whisperx` | WhisperX (torch) | word alignment always on; speaker diarization when `HF_TOKEN` is set; returns all segments at the end |
| `fake` | none | deterministic output without a model, for testing a gateway |

When the gateway asks for a legacy output format, the worker also renders
the transcript exactly as whisper-asr-webservice would return it from `/asr`
for the same engine: it makes the same library calls and uses the same
writer code. Library versions are pinned in `requirements/constraints.txt`
to those of the reference build that whisper-hub is compared with (see
[Compatibility](../README.md#compatibility)).

## Images

CI publishes `ghcr.io/grbit/whisper-hub-worker` with one tag family per
engine, and `-gpu` variants for the three real engines. The tag scheme is
explained in the [main README](../README.md#tags).

To build locally, from the repository root (one engine per image, selected
with `ENGINE`):

```shell
docker build --build-arg ENGINE=faster_whisper -t whisper-hub-worker:faster_whisper worker
docker build --build-arg ENGINE=openai_whisper -t whisper-hub-worker:openai_whisper worker
docker build --build-arg ENGINE=whisperx       -t whisper-hub-worker:whisperx worker
docker build --build-arg ENGINE=fake           -t whisper-hub-worker:fake worker
# CUDA:
docker build -f worker/Dockerfile.gpu --build-arg ENGINE=faster_whisper -t whisper-hub-worker:faster_whisper-gpu worker
```

`make image-worker ENGINE=<engine>` and `make image-worker-fake` do the
same for the CPU images.

## Run

```shell
docker run --rm -p 9100:9100 -e WHISPER_HUB_ENGINE_MODEL=tiny -v whisper-cache:/root/.cache \
  whisper-hub-worker:faster_whisper
docker run --rm -p 9100:9100 whisper-hub-worker:fake
curl http://localhost:9100/v1/info
```

Then point a gateway at it with
`WHISPER_HUB_HUB_WORKERS='[{url: "http://<host>:9100"}]'`. A complete
example is `deploy/docker-compose.split.yml`.

## Configuration

The worker is configured by environment variables only. An invalid value
stops it at startup with exit status 2. An empty value counts as unset.

Settings that whisper-asr-webservice also has keep its meaning; its own
variable names are accepted as aliases, listed in the
[drop-in table](../README.md#migrating-from-whisper-asr-webservice), and the
`WHISPER_HUB_` name wins when both are set:

| Variable | Default | Notes |
|---|---|---|
| `WHISPER_HUB_ENGINE` | set by the image | `faster_whisper`, `whisperx`, `openai_whisper`, `fake`. Outside an image the default is `openai_whisper`, as upstream |
| `WHISPER_HUB_ENGINE_MODEL` | `base` | a model name, a local path, a Hugging Face repository id, or a `https://huggingface.co/...` URL. URLs work with faster_whisper and whisperx only |
| `WHISPER_HUB_ENGINE_MODEL_PATH` | `~/.cache/whisper` | where models are downloaded to. whisperx uses it only for Hugging Face URLs, as upstream |
| `WHISPER_HUB_ENGINE_DEVICE` | `cuda` if available, else `cpu` | openai_whisper ignores it and uses CUDA whenever it is available, as upstream |
| `WHISPER_HUB_ENGINE_QUANTIZATION` | `float32` with CUDA, else `int8` | a CTranslate2 compute type: `float32`, `float16`, `int8`, `bfloat16`, `int16`, `int8_float32`, `int8_float16`, `int8_bfloat16` |
| `WHISPER_HUB_ENGINE_MODEL_IDLE_TIMEOUT` | `0` | seconds without use after which a model is unloaded; 0 means never |
| `HF_TOKEN` | empty | Hugging Face token for downloads; setting it enables whisperx diarization |
| `WHISPER_HUB_ENGINE_SUBTITLE_MAX_LINE_WIDTH` | `1000` | whisperx subtitle output only |
| `WHISPER_HUB_ENGINE_SUBTITLE_MAX_LINE_COUNT` | `2` | whisperx subtitle output only |
| `WHISPER_HUB_ENGINE_SUBTITLE_HIGHLIGHT_WORDS` | `false` | whisperx subtitle output only; only the literal `true` turns it on |
| `WHISPER_HUB_LOG_LEVEL` | `INFO` | `CRITICAL`, `ERROR`, `WARNING` (or `WARN`), `INFO`, `DEBUG`, in any case; the same variable sets the gateway's `log.level`. `DEBUG` logs every request, queueing step and segment |

Variables of the worker itself:

| Variable | Default | Notes |
|---|---|---|
| `WHISPER_HUB_ENGINE_HOST` | `0.0.0.0` | listen address |
| `WHISPER_HUB_ENGINE_PORT` | `9100` | listen port |
| `WHISPER_HUB_ENGINE_MODELS` | the value of `WHISPER_HUB_ENGINE_MODEL` | comma separated list of the models this worker serves; the first is its default model |
| `WHISPER_HUB_ENGINE_MAX_LOADED_MODELS` | `1` | how many models may be in memory at once |
| `WHISPER_HUB_ENGINE_PRELOAD` | `false` | `true` loads the default model at startup instead of on the first job; the worker then starts listening only once the model is loaded |

How models and jobs are handled:

- A model is loaded on the first job that needs it, unless preloaded. When
  the limit of loaded models is reached, the idle model that was used least
  recently is unloaded to make room.
- Each model runs one job at a time; further jobs for it wait in order.
- When the gateway drops the connection (its client went away), the job
  stops: between segments for faster_whisper, between 30-second windows for
  openai_whisper, between the transcribe, align and diarize stages for
  whisperx.

## Fake engine

The fake engine produces deterministic output without a model. It exists so
that a gateway, and anything in front of it, can be tested quickly.

Segment `i` (counting from 0) has `id` i+1, `seek` i*200, `start` 2i, `end`
2i+2, `tokens` `[50364+i]`, `avg_logprob` -0.25, `compression_ratio` 1.2,
`no_speech_prob` 0.01, `temperature` 0.0, and one of three texts in turn:

1. `" Hello from the fake engine."`
2. `" This is segment number two."`
3. `" The quick brown fox jumps over the lazy dog."`

- Language: the requested language with probability 1.0, otherwise
  `WHISPER_HUB_ENGINE_FAKE_LANGUAGE` (default `en`) with probability 0.99. The task is
  ignored. `/v1/detect-language` returns `WHISPER_HUB_ENGINE_FAKE_LANGUAGE` with
  confidence 0.99.
- `word_timestamps`: the words of the text, split on spaces (each with a
  leading space, as Whisper produces them), spread evenly over the segment,
  with probability 0.9. Otherwise `words` is null.
- `legacy_output`: rendered exactly as upstream's faster_whisper engine
  would render these segments.
- `WHISPER_HUB_ENGINE_FAKE_SEGMENTS` (default 3) sets the number of segments.
  `WHISPER_HUB_ENGINE_FAKE_DELAY` (default 0) is the number of seconds waited before
  each segment; the wait ends at once when the client disconnects.
- A `prompt` that starts with `fake:` carries directives instead of a
  prompt, separated by commas or spaces:

  | Directive | Effect |
  |---|---|
  | `delay=<seconds>` | overrides `WHISPER_HUB_ENGINE_FAKE_DELAY` for this job |
  | `segments=<n>` | overrides `WHISPER_HUB_ENGINE_FAKE_SEGMENTS` for this job |
  | `fail=<code>` | fails with an HTTP error before any output; code is `bad_request`, `model_not_found`, `unavailable` or `internal` |
  | `fail_after=<n>` | sends n segments, then an `error` line with code `internal` |

  Example: `fake:delay=0.5,segments=20`. Through the gateway this is the
  legacy parameter `initial_prompt` or the OpenAI field `prompt`.

## Tests

The test suite runs inside a test image, so nothing is installed on the
host:

```shell
docker build --build-arg ENGINE=fake --target test -t whisper-hub-worker:fake-test worker
docker run --rm whisper-hub-worker:fake-test
```

Built with `ENGINE=faster_whisper`, the test image additionally checks the
copied writer code against the installed library.

`scripts/fidelity.py` compares the worker's legacy output with a running
whisper-asr-webservice. Both get the same raw PCM files (16 kHz mono signed
16-bit; upstream receives them with `encode=false`), so audio decoding
cannot cause a difference:

```shell
python3 worker/scripts/fidelity.py --ref http://127.0.0.1:9000 --worker http://127.0.0.1:9100 a.pcm b.pcm
```

It exits with status 1 when any response body differs.
