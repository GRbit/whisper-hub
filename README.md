# whisper-hub

A self-hosted speech-to-text service that speaks three client protocols at
once, in front of Whisper engine workers:

- **Legacy**: the HTTP API of
  [whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice)
  (`/asr`, `/detect-language`, `/docs`). Byte-for-byte compatible: on the same
  uploads, our image and the upstream image return identical status codes,
  headers and bodies (see [Compatibility](#compatibility)).
- **OpenAI**: `/v1/audio/transcriptions`, `/v1/audio/translations`,
  `/v1/models`, at the level of `whisper-1`. Official SDKs work unchanged.
- **Wyoming**: the TCP protocol of Home Assistant voice pipelines (port 10300).

Built in: authentication (HTTP Basic, API keys), per-user model access, usage
accounting, and a priority queue per worker, so Wyoming voice commands do not
wait behind hour-long files.

```
clients --> gateway (Go)                          --> engine worker(s) (Python)
            :9000  legacy + OpenAI + /healthz          :9100 internal HTTP
            :10300 Wyoming                             faster_whisper | whisperx | openai_whisper
            auth, model ACL, usage, ffmpeg decode
```

Audio is decoded once, in the gateway (ffmpeg, the same build upstream
ships, to 16 kHz mono PCM); workers only run models.

## Quick start

All-in-one image, a drop-in for the upstream image: same port, same
environment variables and flags, same model cache path.

```bash
make image                                   # whisper-hub:faster_whisper
docker run -d -p 9000:9000 -p 10300:10300 -e ASR_MODEL=base \
  -v whisper-cache:/root/.cache whisper-hub:faster_whisper
curl -F "audio_file=@meeting.mp3" "http://localhost:9000/asr?output=txt"
```

Or with compose: `docker compose -f deploy/docker-compose.yml up -d`.

Images (everything builds inside docker, no host toolchain needed):

| Image | Make target | Size | Contents |
|---|---|---|---|
| `whisper-hub:faster_whisper` | `make image` | 0.8 GB | gateway + faster_whisper worker (no torch) |
| `whisper-hub:<engine>` | `make image ENGINE=whisperx` | larger (torch) | gateway + whisperx or openai_whisper worker |
| `whisper-hub:<engine>-gpu` | `make image-gpu` | | CUDA variant, **not tested** (no GPU available) |
| `whisper-hub-gateway:latest` | `make image-gateway` | 150 MB | gateway only, uid 65532 |
| `whisper-hub-worker:<engine>` | `make image-worker` | 0.8 GB (faster_whisper) | worker only, see [worker/README.md](worker/README.md) |

Split setup (gateway and workers in separate containers, any number of
workers and engines): `docker compose -f deploy/docker-compose.split.yml up -d`.

### The all-in-one container

The gateway is PID 1 and starts the worker as its child. It forwards
nothing to a shell: on SIGTERM it stops accepting, lets running jobs finish
(`http.shutdown_grace`, 30s; set `stop_grace_period` longer), then stops the
worker; if the worker dies, the gateway exits so the container is
restarted. One process tree, no supervisor script.

- `ASR_ENGINE` unset means the image's own engine. **Deviation:** upstream
  defaults to `openai_whisper`; here the image decides. Asking an image for
  an engine it lacks (e.g. `ASR_ENGINE=openai_whisper` on
  `whisper-hub:faster_whisper`) fails at startup with a message naming the
  image to use.
- The model loads at startup, as upstream. Unlike upstream, the gateway
  listens at once and answers 503 (`/healthz`, `/asr` with
  `{"detail": ...}`) until the model is ready.
- Runs as root, like upstream, so existing root-owned cache volumes on
  `/root/.cache` keep working.
- The worker listens on `127.0.0.1:19100` inside the container; nothing to
  publish.

## Using it

### Legacy API (whisper-asr-webservice)

```bash
curl -F "audio_file=@meeting.mp3" "http://localhost:9000/asr?task=transcribe&output=srt&language=en"
curl -F "audio_file=@meeting.mp3" "http://localhost:9000/detect-language"
```

Same parameters as upstream (`task`, `language`, `initial_prompt`,
`vad_filter`, `word_timestamps`, `diarize`, `min_speakers`, `max_speakers`,
`output=txt|vtt|srt|tsv|json`, `encode`), same validation errors, same
`Asr-Engine` and `Content-Disposition` headers, plus the
`X-Audio-Duration-Seconds` header of the sidecar branch. Options the engine
does not support are ignored, as upstream does. `/docs` and `/openapi.json`
show the schema of the default model's engine.

### OpenAI API

```bash
curl http://localhost:9000/v1/audio/transcriptions \
  -H "Authorization: Bearer $KEY" \
  -F file=@meeting.mp3 -F model=whisper-1 -F response_format=verbose_json
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:9000/v1", api_key="wh_...")
print(client.audio.transcriptions.create(model="whisper-1", file=open("a.mp3", "rb")).text)
```

`response_format` json, text, srt, vtt, verbose_json (with
`timestamp_granularities` word and/or segment); `stream=true` sends
`transcript.text.delta` events as segments are produced. `/v1/models` lists
what the caller may use, aliases included. Unknown model names fall back to
the default model (OpenAI clients hardcode names) unless
`hub.unknown_model: error`.

### Wyoming (Home Assistant)

In Home Assistant: *Settings -> Devices & services -> Add integration ->
Wyoming Protocol*, host = the server's address, port = `10300`. The service
then appears as a speech-to-text engine for Assist pipelines. Zeroconf
discovery (`wyoming.zeroconf.enabled: true`) only works with host
networking, on the same network segment as Home Assistant.

Wyoming carries no credentials: with authentication on, connections act as
the user in `auth.wyoming.user`, optionally limited to source addresses in
`auth.wyoming.allow_cidrs`; a refused connection gets a Wyoming `error`
event.

Verified with the `wyoming` Python package (1.11.0) and a replay of Home
Assistant's STT client code; **not verified against a live Home Assistant
instance.**

### Health and usage

- `GET /healthz` (public): `200 {"status":"ok",...}` once the default model
  is served by a healthy worker, `503` otherwise; reports `workers_up`,
  `workers_total` and `queued`. `whisper-hub healthcheck` does the same from
  inside a container (used by the images' `HEALTHCHECK`).
- `GET /_hub/usage` (admin users, `?user=<name>` to narrow it): per-user
  totals of requests and audio seconds, by protocol and model.

## Configuration

Zero configuration works. Settings come from, in increasing precedence:
built-in defaults, one YAML file (`--config file` or `WHISPER_HUB_CONFIG`),
environment variables, command line flags. Every setting with its default
is in [`deploy/whisper-hub.example.yaml`](deploy/whisper-hub.example.yaml);
`whisper-hub config` prints the effective configuration.

Every setting has an environment variable: `WHISPER_HUB_` plus its YAML
path in upper case, joined by `_`:

| YAML | Environment |
|---|---|
| `http.listen` | `WHISPER_HUB_HTTP_LISTEN=:9000` |
| `hub.default_model` | `WHISPER_HUB_HUB_DEFAULT_MODEL=small` |
| `hub.workers` | `WHISPER_HUB_HUB_WORKERS='[{url: "http://w1:9100"}, {url: "http://w2:9100"}]'` (YAML flow) |
| `hub.aliases` | `WHISPER_HUB_HUB_ALIASES=whisper-1=small,best=large-v3` |
| `auth.users_file` | `WHISPER_HUB_AUTH_USERS_FILE=/config/users.yaml` |
| `auth.protected_paths` | `WHISPER_HUB_AUTH_PROTECTED_PATHS=/asr,/v1/` (comma list) |
| `wyoming.enabled` | `WHISPER_HUB_WYOMING_ENABLED=false` |

Sections: `http` (listeners, shutdown grace), `log` (`level`
debug/info/warn/error, `format` text/json), `audio` (ffmpeg, spool
directory), `hub` (workers, default model, aliases, queue, timeouts),
`auth` (users file, protected paths, rate limit, Wyoming identity, usage
files), `legacy`, `openai`, `wyoming`, `worker` (the child process of the
all-in-one image).

Worker settings (`ASR_MODEL`, `ASR_QUANTIZATION`, `WORKER_MODELS`, ...) are
in [worker/README.md](worker/README.md); in the all-in-one image they reach
the worker unchanged.

### Workers, models and scheduling

Each worker reports its engine and models. Several workers may serve the
same model id; jobs are spread between them. Each worker gets at most
`max_in_flight` jobs at once (default 1); the rest wait in one queue
(`hub.queue_length`, default 100) ordered by priority, FIFO within a
priority. Wyoming is interactive, legacy and OpenAI are normal: a voice
command waits only for the job already running. A waiting job gains one
priority level per `hub.priority_aging` (default 5m), so normal jobs cannot
starve. When the queue is full a new job pushes out the newest
lower-priority waiting job, or is refused with "busy".

## Users and model access

Authentication is off until `auth.users_file` is set (compose:
`deploy/docker-compose.auth.yml`, template in `deploy/auth/`):

```yaml
users:
  - name: alice
    password_hash: "$2a$12$..."    # whisper-hub hash
    api_keys: ["sha256:..."]       # whisper-hub apikey
    models: ["small", "large-*"]   # besides the default model
  - name: homeassistant            # Wyoming identity, no credentials needed
  - name: admin
    api_keys: ["sha256:..."]
    admin: true                    # may read /_hub/usage
```

Create credentials with the binary, inside the container:

```bash
docker compose -f deploy/docker-compose.yml exec whisper whisper-hub hash     # password, no echo
docker compose -f deploy/docker-compose.yml exec whisper whisper-hub apikey   # key for the user + hash for the file
```

- Credentials: HTTP Basic, `Authorization: Bearer <key>`, or `X-Api-Key: <key>`.
- Protected: `/asr`, `/detect-language`, `/v1/*`. Public: `/`, `/docs`,
  `/openapi.json`, `/assets/*`, `/healthz`.
- Every user may use the default model; `models` patterns (exact ids or `*`
  globs) grant more. A known model the user may not use is a 403, never a
  silent fallback. `/v1/models` and Wyoming `info` list only what the caller
  may use.
- Edits apply within 5 seconds; `SIGHUP` reloads at once. A broken edit
  keeps the previous users. A missing file is created as an empty template
  that rejects everyone. Failed logins are rate limited per address.
- `auth.usage.totals_file` keeps per-user totals across restarts;
  `auth.usage.log_file` gets one JSON line per job and rejected login.

## Migrating from whisper-asr-webservice

Replace the image; keep the environment, flags, port and cache volume:

| Upstream | whisper-hub |
|---|---|
| `--host`/`-h`, `--port`/`-p` | the HTTP listener; the image's command may still be `whisper-asr-webservice ...` |
| `ASR_ENGINE` | must match the image's engine (unset = the image's) |
| `ASR_MODEL`, `ASR_MODEL_PATH`, `ASR_DEVICE`, `ASR_QUANTIZATION`, `MODEL_IDLE_TIMEOUT`, `HF_TOKEN`, `SUBTITLE_*` | passed to the worker, same meaning |
| `LOG_LEVEL` | both processes (`WARNING` -> warn, `CRITICAL` -> error) |
| `UPLOAD_SPOOL_DIR`, `DEBUG_FAILED_UPLOADS_DIR` | gateway ffmpeg spool and failed-upload dumps, same meaning |
| `SAMPLE_RATE` | `legacy.raw_sample_rate` (rate of `encode=false` uploads) |
| `/root/.cache` volume | same path, reused |

### From the sidecar sidecar

`SIDECAR_*` variables count only with `SIDECAR_ENABLED=true`, as
before, and reproduce the bundled container: authenticated port
`SIDECAR_LISTEN` (`:9001`), open webservice port
`WHISPER_HOST`:`WHISPER_PORT` (`:9000`, which `--host`/`--port` then
address). Publish only 9001 to require authentication for everyone.

| sidecar | whisper-hub |
|---|---|
| `SIDECAR_CONFIG` (`/config/auth.yaml`) | `auth.users_file`; the old file loads as is |
| `SIDECAR_LISTEN` | `http.listen` |
| `WHISPER_HOST`, `WHISPER_PORT` | `http.open_listen`, the unauthenticated port |
| `SIDECAR_CONFIG_POLL` | `auth.poll_interval` |
| `SIDECAR_USAGE_DIR` (`/data/sidecar`) | `usage.json` there (old totals load as is) and `usage.jsonl` (new lines carry protocol, model, status; old and new shapes then share the file) |
| `SIDECAR_SNAPSHOT_INTERVAL` | `auth.usage.snapshot_interval` |
| `SIDECAR_REALM` | `auth.realm` |
| `SIDECAR_MAX_UPLOAD_BYTES` | `legacy.max_upload_bytes` and `openai.max_upload_bytes` |
| `SIDECAR_TRUST_FORWARDED` | `auth.trust_forwarded` |
| `SIDECAR_LOG_LEVEL` | `log.level`, unless `LOG_LEVEL` is set |
| `SIDECAR_UPSTREAM`, `SIDECAR_BACKEND_WAIT` | no equivalent: no proxy hop, and worker readiness is tracked internally |

The 401 on legacy paths is byte-identical to the old proxy's;
`/_sidecar/healthz` is now `/healthz`.

## Compatibility

What was verified, with the scripts in `deploy/e2e/` (`make e2e`):

- **Legacy, byte for byte** (`make e2e-fidelity`): the reference image
  (upstream + upstream's sidecar branch, faster_whisper, `tiny`, int8)
  against our all-in-one image, the same original uploads (wav, mp3, ogg,
  m4a with a trailing moov atom) with `encode=true`, every output format,
  with options default, `word_timestamps`, `vad_filter`, `language`,
  `task=translate`, `initial_prompt`, plus `/detect-language`: 155 of 155
  responses identical in status, headers and body.
- **Real clients**: the openai Python SDK, the `wyoming` package with a Home
  Assistant STT replay and zeroconf discovery, and the request shapes of
  existing whisper-asr-webservice clients (`POST
  /asr?task=transcribe&output=txt&encode=true[&language=..][&vad_filter=true]`,
  `output=json` reading `text`), with and without Basic auth through the old
  `SIDECAR_*` settings; with a fake engine and with a real `tiny` model.
- **Not verified here**: GPU images, the whisperx and openai_whisper engines
  through the gateway (their images were not built in this round),
  diarization, a live Home Assistant instance, a recording
  actually made by Android MediaMuxer (an ffmpeg-made m4a with the same
  trailing-moov layout was used).

### Known differences from upstream

Legacy API:

- Ready before the model: 503 with `{"detail": ...}` while the worker loads,
  where upstream does not listen yet. `/healthz` is new.
- No `Server: uvicorn` header; header names in canonical case.
- `ASR_ENGINE` unset means the image's engine, not `openai_whisper`.
- whisperx with `initial_prompt` works; upstream answers 500 (its whisperx
  version rejects the argument).
- Kept for fidelity although odd: whisperx diarization passes
  `min_speakers`/`max_speakers` positionally, as upstream does, so
  `min_speakers` lands in pyannote's `num_speakers`; openai_whisper's
  `/detect-language` confidence is the probability of `zh`, not of the
  detected language, as upstream computes it.
- `Asr-Engine` names the engine that ran (several workers may differ).
- Additive: a `model` query parameter, 413 for `legacy.max_upload_bytes`,
  401/403/404/503 for authentication, access and availability.
- HTTP edge cases follow Go's net/http, not uvicorn: paths like `//asr` get a
  301 instead of a 404, the `Allow` header order is fixed, request limits
  and malformed HTTP handling differ; speaker counts beyond the int range
  saturate; multipart charsets other than utf-8 and latin-1 read as utf-8.
- The worker allowlists the classes of the pyannote VAD checkpoint bundled
  with whisperx for torch's safe loading; the reference image fails to start
  whisperx for that reason.

OpenAI API (compared with `whisper-1`):

- No 25 MB limit by default (`openai.max_upload_bytes`).
- Unknown model names fall back to the default model.
- `diarized_json` is rejected (no speaker diarization for whisper models);
  gpt-only fields (`chunking_strategy`, `include`) are accepted and ignored.
- `usage` reports whole seconds, rounded up; `owned_by` is `whisper-hub`.
- An engine failure in the middle of a stream is reported with the
  gateway's own message, not the engine's.

Wyoming:

- Model ids are listed, aliases are not; the program name is `whisper-hub`.
- `zh-HK` is advertised for Cantonese (`yue`), as wyoming-faster-whisper
  does, because Home Assistant cannot select `yue` otherwise.
- Transcript streaming events are off by default
  (`wyoming.stream_transcripts`): Home Assistant reads only the final
  transcript.

## Operations

- Logs go to stderr: one Info line per job (the usage record), Debug for
  every step. The worker logs its own lines into the same stream.
- `SIGTERM`/`SIGINT`: stop accepting, let running requests finish for
  `http.shutdown_grace` (30s), then cut them off (which also stops their
  worker jobs), stop the managed worker, write the final usage snapshot.
- `SIGHUP`: reload the users file, reopen the request log.

## Development

```bash
make check         # gofmt, go vet, go test -race ./...
make test-short    # skips the slow bcrypt tests
make e2e           # all end-to-end runs below (docker; builds the images)
make e2e-clients   # owners' openai SDK / wyoming / HA / zeroconf scripts, canned fake worker
make e2e-fake      # gateway + Python fake worker: protocols, streaming, cancellation, priority, auth
make e2e-real      # all-in-one, model tiny, real audio; sidecar compatibility mode
make e2e-fidelity  # byte-for-byte legacy comparison with the reference image
```

Design and conventions: [`docs/DESIGN.md`](docs/DESIGN.md).
