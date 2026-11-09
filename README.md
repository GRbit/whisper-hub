# whisper-hub

A self-hosted speech-to-text gateway. One Go binary speaks three client
protocols at once and sends the work to Python Whisper engine workers:

- **Legacy**: the HTTP API of
  [whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice)
  (`/asr`, `/detect-language`, `/docs`), compatible down to status codes,
  headers and output bytes, so existing clients keep working.
- **OpenAI**: `/v1/audio/transcriptions`, `/v1/audio/translations`,
  `/v1/models`, at the level of `whisper-1`. Official SDKs work unchanged.
- **Wyoming**: the TCP protocol of Home Assistant voice pipelines (port 10300).

Built in: authentication (HTTP Basic, API keys), per-user model access, usage
accounting, a priority queue per worker (Wyoming voice commands do not wait
behind hour-long files).

```
clients --> whisper-hub (Go)                      --> engine worker(s) (Python)
            :9000  legacy + OpenAI + /healthz          :9100 internal HTTP
            :10300 Wyoming                             faster-whisper | whisperx | openai-whisper
            auth, model ACL, usage, ffmpeg decode
```

Audio is decoded once, in the gateway (ffmpeg to 16 kHz mono PCM); workers
only run models.

> **Status.** The gateway is complete. Still pending (phase 2): the engine
> worker image, the all-in-one image (gateway + faster-whisper worker, a
> drop-in for the upstream image), docker compose files, and the end-to-end
> comparison with the upstream service on a real model. Sections below that
> depend on those are marked *pending*.

## Quick start

Build and run the gateway (Go 1.25+, ffmpeg in `PATH`):

```bash
make build
bin/whisper-hub            # HTTP :9000, Wyoming :10300, worker http://127.0.0.1:9100, no auth
```

With no worker running yet the gateway still starts; `/healthz` answers 503
until a worker reports its models, and requests get "unavailable".

Gateway image (ffmpeg included, runs as uid 65532):

```bash
make image-gateway
docker run -p 9000:9000 -p 10300:10300 \
  -e 'WHISPER_HUB_HUB_WORKERS=[{url: "http://my-worker:9100"}]' \
  whisper-hub-gateway:latest
```

Worker image and all-in-one image: *pending (phase 2)*.

## Using it

### Legacy API (whisper-asr-webservice)

```bash
curl -F "audio_file=@meeting.mp3" "http://localhost:9000/asr?output=srt&language=en"
curl -F "audio_file=@meeting.mp3" "http://localhost:9000/detect-language"
```

Same parameters as upstream (`task`, `language`, `initial_prompt`,
`vad_filter`, `word_timestamps`, `diarize`, `min_speakers`, `max_speakers`,
`output=txt|vtt|srt|tsv|json`, `encode`), same validation errors, same
`Asr-Engine` and `Content-Disposition` headers. Options the engine does not
support are ignored, as upstream does. `/docs` and `/openapi.json` show the
schema of the default model's engine. While no worker is up, `/openapi.json`
and `/asr` answer 503 with `{"detail": ...}`.

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
the models the caller may use, aliases included. Unknown model names fall
back to the default model (OpenAI clients often hardcode names) unless
`hub.unknown_model: error`.

### Wyoming (Home Assistant)

In Home Assistant: *Settings -> Devices & services -> Add integration ->
Wyoming Protocol*, host = the gateway's address, port = `10300`. The
gateway then appears as a speech-to-text engine for Assist pipelines.
Zeroconf discovery (`wyoming.zeroconf.enabled: true`) works only when the
gateway shares a network segment with Home Assistant (host networking).

Wyoming carries no credentials: with authentication on, connections act as
the user named in `auth.wyoming.user`, optionally limited to source
addresses in `auth.wyoming.allow_cidrs`. A refused connection gets a
Wyoming `error` event, which Home Assistant shows.

### Health and usage

- `GET /healthz` (public): `200 {"status":"ok",...}` once the default model
  is served by a healthy worker, `503` otherwise; also reports
  `workers_up`, `workers_total` and `queued`.
- `GET /_hub/usage` (admin users only, `?user=<name>` to narrow it): per-user
  totals of requests and audio seconds, by protocol and by model.

## Configuration

Zero configuration works. Settings come from, in increasing precedence:
built-in defaults, one YAML file (`--config file` or `WHISPER_HUB_CONFIG`),
environment variables, command line flags. The commented reference with
every setting and its default is
[`deploy/whisper-hub.example.yaml`](deploy/whisper-hub.example.yaml).

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

`whisper-hub config` prints the effective configuration (zero values mean
the component's default).

Main sections: `http` (listeners, shutdown grace), `log` (`level`
debug/info/warn/error, `format` text/json), `audio` (ffmpeg, spool
directory), `hub` (workers, default model, aliases, queue, timeouts),
`auth` (users file, protected paths, rate limit, Wyoming identity, usage
files), `legacy`, `openai`, `wyoming`.

### Workers, models and scheduling

Each worker reports its engine and models. Several workers may serve the
same model id; jobs are spread between them. Each worker gets at most
`max_in_flight` jobs at once (default 1); the rest wait in one queue
(`hub.queue_length`, default 100) ordered by priority (Wyoming interactive
first), FIFO within a priority. A waiting job gains one priority level per
`hub.priority_aging` (default 5m), so batch jobs cannot starve. When the
queue is full a new job pushes out the newest lower-priority waiting job, or
is refused with "busy".

## Users and model access

Authentication is off until `auth.users_file` is set. The file:

```yaml
users:
  - name: alice
    password_hash: "$2a$12$..."    # whisper-hub hash
    api_keys: ["sha256:..."]       # whisper-hub apikey
    models: ["small", "large-*"]   # besides the default model
  - name: homeassistant            # Wyoming identity, no credentials needed
    models: ["*"]
  - name: admin
    api_keys: ["sha256:..."]
    admin: true                    # may read /_hub/usage
  - name: mallory
    password_hash: "$2a$12$..."
    disabled: true
```

```bash
whisper-hub hash      # asks for a password (no echo), prints the bcrypt hash
whisper-hub apikey    # prints a new key (give it to the user) and its hash (store it)
```

- Credentials: HTTP Basic, `Authorization: Bearer <key>`, or `X-Api-Key: <key>`.
- Protected: `/asr`, `/detect-language`, `/v1/*`. Public: `/`, `/docs`,
  `/openapi.json`, `/assets/*`, `/healthz`.
- Every user may use the default model; `models` patterns (exact ids or `*`
  globs) grant more. A request for a known model the user may not use fails
  with 403, never a silent fallback. `/v1/models` and Wyoming `info` list
  only what the caller may use.
- Edits to the file apply within `auth.poll_interval` (5s); `SIGHUP`
  reloads at once. A broken edit keeps the previous users in force. A
  missing file is created as an empty template that rejects everyone.
- Failed logins are rate limited per client address (`auth.rate_limit`).
- `auth.usage.totals_file` keeps per-user totals across restarts,
  `auth.usage.log_file` gets one JSON line per job and rejected login.

## Migrating from whisper-asr-webservice

The gateway accepts what an upstream deployment passes:

| Upstream | whisper-hub |
|---|---|
| `--host`/`-h`, `--port`/`-p` | the HTTP listener (`http.listen`) |
| `LOG_LEVEL` | `log.level` (`WARNING` -> warn, `CRITICAL` -> error) |
| `UPLOAD_SPOOL_DIR` | `audio.spool_dir` |
| `DEBUG_FAILED_UPLOADS_DIR` | `audio.failed_uploads_dir` |
| `SAMPLE_RATE` | `legacy.raw_sample_rate` (rate of `encode=false` uploads) |
| `ASR_ENGINE`, `ASR_MODEL`, `ASR_MODEL_PATH`, `ASR_DEVICE`, `ASR_QUANTIZATION`, `MODEL_IDLE_TIMEOUT`, `HF_TOKEN`, `SUBTITLE_*` | engine worker settings; the gateway ignores them, the all-in-one image passes them to its worker (*pending*) |

Differences: the gateway is ready before the model is (it answers 503 until
the worker is up, where upstream did not listen at all); `/healthz` is new.

### From the sidecar sidecar

`SIDECAR_*` variables count only with `SIDECAR_ENABLED=true`, as
before. That mode reproduces the bundled container: the authenticated
listener on `SIDECAR_LISTEN` (default `:9001`) and the open webservice
port on `WHISPER_HOST`:`WHISPER_PORT` (default `:9000`), which the upstream
`--host`/`--port` flags then address. Publish only 9001 to require
authentication for everyone.

| sidecar | whisper-hub |
|---|---|
| `SIDECAR_CONFIG` (`/config/auth.yaml`) | `auth.users_file`; the old file loads as is (`name`, `password_hash`) |
| `SIDECAR_LISTEN` (`:9001`) | `http.listen` |
| `WHISPER_HOST`, `WHISPER_PORT` | `http.open_listen`, the unauthenticated port |
| `SIDECAR_CONFIG_POLL` | `auth.poll_interval` |
| `SIDECAR_USAGE_DIR` (`/data/sidecar`) | `auth.usage.totals_file` = `<dir>/usage.json` (old totals load as is), `auth.usage.log_file` = `<dir>/usage.jsonl` (new lines have a different shape: protocol, model, status) |
| `SIDECAR_SNAPSHOT_INTERVAL` | `auth.usage.snapshot_interval` |
| `SIDECAR_REALM` | `auth.realm` |
| `SIDECAR_MAX_UPLOAD_BYTES` | `legacy.max_upload_bytes` and `openai.max_upload_bytes` |
| `SIDECAR_TRUST_FORWARDED` | `auth.trust_forwarded` |
| `SIDECAR_LOG_LEVEL` | `log.level`, unless `LOG_LEVEL` is set |
| `SIDECAR_UPSTREAM`, `SIDECAR_BACKEND_WAIT` | no equivalent: there is no proxy hop, and the gateway tracks worker readiness itself |

The 401 on legacy paths is byte-identical to the old proxy's (Basic
challenge, `unauthorized`); `/_sidecar/healthz` is now `/healthz`.

## Operations

- Logs go to stderr; one Info line per job, Debug for every step.
- `SIGTERM`/`SIGINT`: stop accepting, let running requests finish for
  `http.shutdown_grace` (30s), then cut them off (which also stops the worker
  jobs) and write the final usage snapshot. Give docker a longer
  `stop_grace_period`.
- `SIGHUP`: reload the users file and reopen the request log.

## Development

```bash
make check         # gofmt, go vet, go test -race ./...
make test-short    # skips the slow bcrypt tests
make e2e-docker    # openai SDK, wyoming package and a Home Assistant replay
                   # against the gateway image with a fake worker
```

Design and conventions: [`docs/DESIGN.md`](docs/DESIGN.md).
