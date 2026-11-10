# whisper-hub

A self-hosted speech-to-text service that speaks three client protocols at
once, in front of [Whisper](https://github.com/openai/whisper) models:

- **Legacy**: the HTTP API of
  [whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice)
  (`/asr`, `/detect-language`, `/docs`), called "upstream" in this document.
  whisper-hub is a drop-in replacement for it: same port, same environment
  variables and flags, same model cache, and the same API, with a few
  listed extensions (see [Compatibility](#compatibility)).
- **OpenAI**: `/v1/audio/transcriptions`, `/v1/audio/translations` and
  `/v1/models`, at the level of OpenAI's `whisper-1` model. The official
  SDKs work unchanged.
- **Wyoming**: the TCP protocol of Home Assistant voice pipelines, on port
  10300.

Built in: authentication (HTTP Basic and API keys), per-user model access,
usage accounting (how much audio each user had transcribed), and a priority
queue, so that a voice command does not wait behind an hour-long file.

```
clients --> gateway (Go)                          --> engine worker(s) (Python)
            :9000  legacy + OpenAI + /healthz          :9100 internal HTTP
            :10300 Wyoming                             faster_whisper | whisperx | openai_whisper
            auth, model access, usage, ffmpeg decode
```

The **gateway** is one Go binary that handles clients. A **worker** is a
Python process that runs one **engine** (a Whisper implementation:
faster_whisper, whisperx or openai_whisper) and serves one or more models.
Audio is decoded once, in the gateway; workers only run models. The usual
way to run it is the all-in-one image, which contains both.

Contents: [Quick start](#quick-start) -
[Images and tags](#images-and-tags) - [Configuration](#configuration) -
[Using the protocols](#using-the-protocols) -
[Authentication](#authentication-and-model-access) -
[Usage accounting](#usage-accounting) -
[Migrating](#migrating-from-whisper-asr-webservice) -
[Compatibility](#compatibility) -
[Operations](#operations) - [Development](#development)

## Quick start

Run the all-in-one image and transcribe a file:

```bash
docker run -d --name whisper -p 9000:9000 -p 10300:10300 -e WHISPER_HUB_ENGINE_MODEL=base \
  -v whisper-cache:/root/.cache -v whisper-usage:/data ghcr.io/grbit/whisper-hub:edge
curl -F "audio_file=@meeting.mp3" "http://localhost:9000/asr?output=txt"
```

`meeting.mp3` stands for any audio or video file ffmpeg can read. The first
start downloads the model into the `whisper-cache` volume; until the model
is loaded, requests and `http://localhost:9000/healthz` answer 503. The
`whisper-usage` volume keeps the per-user usage totals (see
[Usage accounting](#usage-accounting)).

The example uses the `edge` tag because no version has been released yet:
release tags such as `latest` appear with the first release. See
[Tags](#tags).

To build the image from a checkout instead of pulling it:

```bash
make image                                        # builds whisper-hub:faster_whisper
docker compose -f deploy/docker-compose.yml up -d # runs it; builds it first if missing
```

Everything is built inside Docker; no Go or Python toolchain is needed on
the host. The compose files refer to locally built image names. To run a
published image with them, change `image:` to the `ghcr.io/grbit/...` name.

## Images and tags

### Images

CI publishes these images to `ghcr.io/grbit/`, for linux/amd64 only:

| Image | Size | Contents |
|---|---|---|
| `whisper-hub:faster_whisper` | 0.8 GB | all-in-one: gateway + faster_whisper worker (no torch). The default variant |
| `whisper-hub:openai_whisper` | 2.7 GB | all-in-one: gateway + openai_whisper worker |
| `whisper-hub:whisperx` | 3.7 GB | all-in-one: gateway + whisperx worker |
| `whisper-hub:all` | 4.0 GB | all-in-one: gateway + all three engines; `WHISPER_HUB_ENGINE` picks one at startup |
| `whisper-hub:<engine>-gpu` | about 7 GB (faster_whisper) to 15 GB (all) | CUDA builds of the four above, see [GPU](#gpu) |
| `whisper-hub-gateway` | 150 MB | gateway only, for the [split setup](#split-setup); runs as uid 65532 |
| `whisper-hub-worker:<engine>` | | worker only, for the split setup: `faster_whisper` (the default variant), `openai_whisper`, `whisperx`, and the model-free `fake` used in tests. See [worker/README.md](worker/README.md) |
| `whisper-hub-worker:<engine>-gpu` | about 7 to 14 GB | CUDA workers: `faster_whisper`, `openai_whisper`, `whisperx` |

### Tags

A tag consists of a channel and a variant. The **variant** is the engine
name from the table above (`faster_whisper`, `whisperx`, `all`,
`whisperx-gpu`, ...). The **channel** says which build you get:

| Channel | Tag form | Example | Points at |
|---|---|---|---|
| edge | `edge-<variant>` | `whisper-hub:edge-whisperx` | the newest commit on the `master` branch. It moves with every push, and is not a release |
| commit | `sha-<commit>-<variant>` | `whisper-hub:sha-a53e998-whisperx` | the build of one `master` commit (first 7 characters of its hash). It never moves |
| version | `<version>-<variant>` | `whisper-hub:1.2.3-whisperx` | the release made from git tag `v1.2.3`. It never moves |
| latest release | `<variant>` | `whisper-hub:whisperx` | the newest release of that variant. It moves with every release |

The default variant of an image also gets the short tags without a variant:

| Short tag | Same as | Exists for |
|---|---|---|
| `edge` | `edge-faster_whisper` | `whisper-hub`, `whisper-hub-worker`, `whisper-hub-gateway` |
| `sha-<commit>` | `sha-<commit>-faster_whisper` | the same three |
| `<version>`, for example `1.2.3` | `1.2.3-faster_whisper` | the same three |
| `latest` | `faster_whisper` | the same three |

The gateway image has no variants, so it has only the short tags. The CUDA
builds have no short tags: there is no `latest-gpu` and no plain `edge` for
them; use `faster_whisper-gpu`, `edge-faster_whisper-gpu` and so on.

Which one to use:

- `edge` (or `edge-<variant>`) to follow development. This is the only
  channel that exists before the first release.
- `latest` (or `<variant>`) to follow releases, once there is one.
- A version tag or a `sha-` tag to pin a deployment to one build.

Tags starting with `buildcache` hold CI's layer cache. They are not
runnable images.

Release tags are created when a `v*` git tag is pushed, edge and commit tags
on every push to `master`. In a published image, `whisper-hub version`
prints the branch or tag name and the full commit hash it was built from.

### Choosing the engine

- **By image**: `whisper-hub:faster_whisper`, `:openai_whisper` or
  `:whisperx`. These are smaller. With `WHISPER_HUB_ENGINE` unset the image
  runs its own engine. This differs from upstream, whose single image defaults to
  `openai_whisper`. Asking for an engine the image does not contain fails at
  startup with a message naming the image to use.
- **By `WHISPER_HUB_ENGINE` on `whisper-hub:all`**: exactly as upstream.
  One image holds every engine and an unset `WHISPER_HUB_ENGINE` means
  `openai_whisper`. One engine runs per container start.

### GPU

The `-gpu` images are built on the NVIDIA CUDA 12.9 runtime and need the
NVIDIA Container Toolkit on the host:

```bash
docker run -d --gpus all -p 9000:9000 -p 10300:10300 \
  -e WHISPER_HUB_ENGINE_MODEL=large-v3 -e WHISPER_HUB_ENGINE_QUANTIZATION=float16 \
  -v whisper-cache:/root/.cache -v whisper-usage:/data ghcr.io/grbit/whisper-hub:edge-faster_whisper-gpu
```

As upstream: `WHISPER_HUB_ENGINE_DEVICE` defaults to `cuda` when a GPU is
visible and to `cpu` otherwise; `WHISPER_HUB_ENGINE_QUANTIZATION` defaults
to `float32` with CUDA and to `int8` without (`float16` or `int8_float16` are the usual choices on a GPU).
`whisper-hub:all-gpu` takes `WHISPER_HUB_ENGINE` like `whisper-hub:all`. With
compose, add `deploy.resources.reservations.devices` with
`capabilities: [gpu]` to the service.

### The all-in-one container

The gateway is the container's main process (PID 1) and starts the worker as
its child. There is one process tree and no supervisor script.

- The model loads at startup, as upstream. Unlike upstream, the gateway
  listens at once and answers 503 until the model is ready (`/healthz`, and
  `/asr` with a `{"detail": ...}` body).
- On `SIGTERM` it stops accepting requests, lets running jobs finish for up
  to `http.shutdown_grace` (30s), then stops the worker. Give Docker a
  longer `stop_grace_period` than that.
- If the worker dies, the gateway exits, so the container is restarted.
- It runs as root, like upstream, so existing root-owned cache volumes on
  `/root/.cache` keep working.
- The worker listens on `127.0.0.1:19100` inside the container. There is
  nothing to publish.

### Split setup

The gateway and the workers can run in separate containers, with any number
of workers and engines:

```bash
docker compose -f deploy/docker-compose.split.yml up -d
```

The gateway finds its workers through `hub.workers`
(`WHISPER_HUB_HUB_WORKERS`). Worker settings are described in
[worker/README.md](worker/README.md). The worker port has no authentication:
keep it on a private network.

## Configuration

No configuration is needed: without any, the gateway serves HTTP on `:9000`
and Wyoming on `:10300`, expects one worker at `http://127.0.0.1:9100`, and
has authentication off.

Settings come from four sources. A later one overrides an earlier one:

1. built-in defaults;
2. one YAML file, named by `--config <file>` or `WHISPER_HUB_CONFIG`;
3. environment variables;
4. the command line flags `--host` and `--port`.

Every setting, with its default and an explanation, is in
[`deploy/whisper-hub.example.yaml`](deploy/whisper-hub.example.yaml). An
unknown key in the file is an error.

Every setting has an environment variable: `WHISPER_HUB_`, then the YAML
path in upper case with `_` between the parts.

| YAML | Environment |
|---|---|
| `http.listen` | `WHISPER_HUB_HTTP_LISTEN=:9000` |
| `hub.default_model` | `WHISPER_HUB_HUB_DEFAULT_MODEL=small` |
| `hub.workers` | `WHISPER_HUB_HUB_WORKERS='[{url: "http://w1:9100"}, {url: "http://w2:9100"}]'` (YAML flow syntax) |
| `hub.aliases` | `WHISPER_HUB_HUB_ALIASES=whisper-1=small,best=large-v3` (`key=value` pairs) |
| `auth.users_file` | `WHISPER_HUB_AUTH_USERS_FILE=/config/users.yaml` |
| `auth.protected_paths` | `WHISPER_HUB_AUTH_PROTECTED_PATHS=/asr,/v1/` (comma separated list) |
| `wyoming.enabled` | `WHISPER_HUB_WYOMING_ENABLED=false` |

The variable names of an upstream deployment are accepted as aliases, see
[Migrating](#migrating-from-whisper-asr-webservice). When both an alias and
a `WHISPER_HUB_` variable set the same thing, the `WHISPER_HUB_` one wins.

The sections of the file:

| Section | Configures |
|---|---|
| `http` | HTTP listeners, shutdown grace period |
| `log` | `level` (debug, info, warn, error) and `format` (text, json) |
| `audio` | ffmpeg path, spool directory for uploads, dumps of uploads that fail to decode |
| `hub` | workers, default model, aliases, queue, timeouts |
| `auth` | users file, protected paths, rate limit, Wyoming source networks, usage files |
| `legacy`, `openai`, `wyoming` | the three protocols |
| `worker` | the worker process the gateway starts itself; the all-in-one images set it |

Worker settings (`WHISPER_HUB_ENGINE`, `WHISPER_HUB_ENGINE_MODEL`,
`WHISPER_HUB_ENGINE_QUANTIZATION`, `WHISPER_HUB_ENGINE_MODELS`, ...) are
environment variables of the worker, listed in
[worker/README.md](worker/README.md). They all start with
`WHISPER_HUB_ENGINE`, a name no gateway setting derives. In the all-in-one
image, set them on the container: the worker inherits its environment.

### Command line

| Command | Does |
|---|---|
| `whisper-hub [serve] [flags]` | runs the gateway. `serve` is the default and may be omitted |
| `whisper-hub whisper-asr-webservice [flags]` | the same, under upstream's command name; the images also have a `whisper-asr-webservice` executable |
| `whisper-hub hash` | asks for a password and prints its hash for the users file |
| `whisper-hub apikey` | prints a new API key and its hash for the users file |
| `whisper-hub config [flags]` | prints the effective configuration: file, environment and flags merged, with every unset setting shown as its default |
| `whisper-hub healthcheck [flags]` | exits 0 when `/healthz` of the gateway in the same container answers 200; the images use it as their `HEALTHCHECK` |
| `whisper-hub version` | prints the version |

Flags: `--config <file>`, `--host <address>` (short `-h`), `--port <port>`
(short `-p`). The short forms are upstream's, so `-h` is the host, not help;
`--help` prints the usage.

### Models, aliases and scheduling

Each worker reports its engine and its models. The gateway offers all of
them under their ids (for example `base`, `large-v3`), plus aliases.

- The **default model** runs requests that name no model. It is
  `hub.default_model`, or, when that is not set, the default model of the
  first healthy worker in `hub.workers`.
- An **alias** (`hub.aliases`) is another name for a model. `whisper-1` is
  an alias of the default model out of the box, because OpenAI clients send
  that name.
- A model name nobody knows falls back to the default model, since many
  OpenAI clients hardcode names. Set `hub.unknown_model: error` to refuse
  such requests instead.

Several workers may serve the same model; jobs are spread between them.
Each worker gets at most `max_in_flight` jobs at once (default 1). The rest
wait in one queue (`hub.queue_length`, default 100), ordered by priority and
by arrival within a priority. Wyoming requests have interactive priority,
legacy and OpenAI requests normal priority: a voice command waits only for
the job already running. Waiting does not raise a job's priority, so a
constant stream of higher-priority jobs keeps lower ones waiting. When the
queue is full, a new job pushes out the newest waiting job of lower
priority, or is itself refused as busy (503 over HTTP).

## Using the protocols

### Legacy API (whisper-asr-webservice)

```bash
curl -F "audio_file=@meeting.mp3" "http://localhost:9000/asr?task=transcribe&output=srt&language=en"
curl -F "audio_file=@meeting.mp3" "http://localhost:9000/detect-language"
```

`POST /asr` takes the file as the multipart field `audio_file` and these
query parameters, all optional, exactly as upstream:

| Parameter | Default | Meaning |
|---|---|---|
| `task` | `transcribe` | `transcribe`, or `translate` (to English) |
| `language` | detect | a Whisper language code such as `en` |
| `output` | `txt` | `txt`, `vtt`, `srt`, `tsv` or `json` |
| `initial_prompt` | none | text that primes the decoder |
| `vad_filter` | `false` | voice activity filter (faster_whisper) |
| `word_timestamps` | `false` | word-level timestamps |
| `diarize`, `min_speakers`, `max_speakers` | off | speaker labels (whisperx with `HF_TOKEN`) |
| `encode` | `true` | `false` means the upload is already raw 16-bit mono PCM |
| `model` | the default model | a whisper-hub addition: a model id or alias |

Validation errors, the `Asr-Engine` and `Content-Disposition` headers and
the response bodies are upstream's. Options the engine does not support are
ignored, as upstream does. `/docs` and `/openapi.json` show the schema of
the default model's engine.

`POST /detect-language` answers
`{"detected_language":"english","language_code":"en","confidence":0.98,"audio_duration_seconds":12.34}`.
It looks at the first 30 seconds.

### OpenAI API

```bash
curl http://localhost:9000/v1/audio/transcriptions \
  -H "Authorization: Bearer $KEY" \
  -F file=@meeting.mp3 -F model=whisper-1 -F response_format=verbose_json
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:9000/v1", api_key="wh_...")
print(client.audio.transcriptions.create(model="whisper-1", file=open("meeting.mp3", "rb")).text)
```

With authentication off, the key can be any text.

| Endpoint | Notes |
|---|---|
| `POST /v1/audio/transcriptions` | fields `file`, `model`, `language`, `prompt`, `temperature`, `response_format` (`json`, `text`, `srt`, `vtt`, `verbose_json`), `timestamp_granularities[]` (`segment`, `word`; for `verbose_json`), `stream` |
| `POST /v1/audio/translations` | the same, translating to English; `language`, `stream` and `timestamp_granularities[]` are ignored |
| `GET /v1/models`, `GET /v1/models/{id}` | the model ids and aliases the caller may use |

With `stream=true` and format `json` or `text`, the response is a stream of
server-sent events: one `transcript.text.delta` per segment as the engine
produces it, then `transcript.text.done`.

### Wyoming (Home Assistant)

In Home Assistant: *Settings -> Devices & services -> Add integration ->
Wyoming Protocol*, host = the server's address, port = `10300`. The service
then appears as a speech-to-text engine for Assist pipelines.

Home Assistant can also discover the service by zeroconf
(`wyoming.zeroconf.enabled: true`). That only works when the container uses
host networking and is on the same network segment as Home Assistant.

Wyoming carries no credentials, so the network decides who may connect:
keep the port on a network you trust, and optionally limit source addresses
with `auth.wyoming.allow_cidrs`. Every Wyoming connection acts as the user
`wyoming`, with authentication on or off, so its jobs and usage have their
own entry. With authentication on, a users file entry named `wyoming`
restricts it: its `models` patterns apply, and `disabled: true` on that
entry makes whisper-hub refuse every Wyoming connection:

```yaml
  - name: wyoming
    disabled: true
```

Without such an entry Wyoming may use every model. A refused connection
gets a Wyoming `error` event.

### Health

`GET /healthz` is always public. It answers
`200 {"queued":0,"status":"ok","workers_total":1,"workers_up":1}` once the
default model is served by a healthy worker, and 503 with the same fields
and `"status":"unavailable"` otherwise. `queued` is the number of jobs
waiting for a worker.

## Authentication and model access

Authentication is off until `auth.users_file` is set. While it is off,
every request has admin access and may use every model, as with upstream:
the service trusts its callers because the network restricts who can reach
it. The same admin access applies, with authentication on, to paths outside
`auth.protected_paths` and to the optional second listener
`http.open_listen` (the admin port), which serves the same routes without
authentication. Model restrictions are for users only: admin access may
always use every model.

To turn it on with compose:

```bash
cp deploy/auth/users.example.yaml deploy/auth/users.yaml   # then edit it
docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.auth.yml up -d
```

The users file:

```yaml
users:
  - name: alice
    password_hash: "$2a$12$..."    # from: whisper-hub hash
    api_keys: ["sha256:..."]       # from: whisper-hub apikey
    models: ["small", "large-*"]   # allowed besides the default model
  - name: wyoming                  # optional: restricts Wyoming connections
    models: ["small"]
  - name: admin
    api_keys: ["sha256:..."]
    admin: true                    # may read /_hub/usage
```

The file holds hashes only, never a password or a key. Create them with the
binary, for example inside the running container:

```bash
docker compose -f deploy/docker-compose.yml exec whisper whisper-hub hash     # asks for the password twice, no echo
docker compose -f deploy/docker-compose.yml exec whisper whisper-hub apikey   # prints the key once, and its hash
```

Give the key (it starts with `wh_`) to the user and put the hash into the
file. A user entry may also have `disabled: true`.

How it behaves:

- **Credentials**: HTTP Basic (name and password),
  `Authorization: Bearer <key>`, or `X-Api-Key: <key>`.
- **Protected paths**: `/asr`, `/detect-language` and everything under
  `/v1/` (`auth.protected_paths`). Public: `/`, `/docs`, `/openapi.json`,
  `/assets/*`, `/healthz`.
- **Rejections**: a legacy path answers 401 with a Basic challenge, so a
  browser asks for a login; a `/v1/` path answers 401 with an OpenAI error
  object.
- **Model access**: every user may use the default model. The `models`
  patterns (exact ids, or globs with `*`) grant more. A request for a model
  the user may not use is refused with 403; it never falls back silently.
  `/v1/models` and the Wyoming model list show only what the caller may use.
- **Editing the file**: changes apply within 5 seconds, and `SIGHUP`
  reloads at once. A file that does not parse keeps the previous users in
  force. A missing file is created as an empty template, which rejects
  everyone until it names a user.
- **Rate limit**: after 10 failed attempts within a minute from one client
  address (`auth.rate_limit`), further attempts from it are rejected with
  401 and a `Retry-After` header until the minute is over. Behind a reverse
  proxy set `auth.trust_forwarded: true`, so that the client address is
  taken from `X-Forwarded-For`.
- **Wyoming**: see [above](#wyoming-home-assistant). An entry named
  `wyoming` is how Wyoming is restricted; the port itself must be protected
  by the network.

## Usage accounting

whisper-hub records how much audio each user had transcribed. The unit is
the **job**: one transcription or language detection. Every job produces one
usage record when it finishes, whether it succeeded or not, with the length
of the decoded audio in seconds. (For language detection that is at most 30
seconds, since only that much is used.) Requests that are rejected before
they become a job, such as a failed login or an upload that is not audio,
produce no usage record.

There are four places to read it:

| Where | On by default | Contains |
|---|---|---|
| Log line `job finished` on stderr | yes | one line per job |
| Request log file | no: set `auth.usage.log_file` | one JSON line per job, and one per rejected login |
| Totals file | in the images: `/data/usage.json`; the bare binary: set `auth.usage.totals_file` | running totals per user, kept across restarts |
| `GET /_hub/usage` | for admin access: always while authentication is off; with it on, on the admin port `http.open_listen` or for a user with `admin: true` | the current totals per user |

Every image that contains the gateway declares `/data` as a volume and
writes the totals to `/data/usage.json` there; mount a named volume on
`/data` so they survive a replaced container. The compose files do (`usage-data`), and the overlay
`deploy/docker-compose.auth.yml` adds the request log `/data/requests.jsonl`
next to the totals. Without compose:

```bash
docker run -d -p 9000:9000 -v whisper-cache:/root/.cache -v whisper-usage:/data \
  -e WHISPER_HUB_AUTH_USAGE_LOG_FILE=/data/requests.jsonl \
  ghcr.io/grbit/whisper-hub:edge
```

**The log line** is written at Info level for every job and needs no
configuration (`docker logs <container>`):

```
time=2026-11-10T12:00:09.120Z level=INFO msg="job finished" user=alice protocol=legacy endpoint=/asr model=base engine=faster_whisper task=transcribe status=ok audio_seconds=83.4 queued_ms=0 elapsed_ms=9120
```

`status` is `ok` or the kind of failure (`forbidden`, `unavailable`,
`canceled`, ...); a failed job also has an `error` field. `queued_ms` is the
time spent waiting for a worker, `elapsed_ms` the total time including it.
With `log.format: json` the line is a JSON object.

**The request log file** has the same fields per job as JSON: `ts`, `user`,
`protocol` (`legacy`, `openai`, `wyoming`), `endpoint`, `model`, `engine`,
`task`, `audio_seconds`, `queued_ms`, `elapsed_ms`, `status`, `error`. A
rejected login on a protected path adds a line with `status`
`unauthenticated`, `attempted_user` (the name of a Basic login), `client_ip`
and `reason` (`missing`, `invalid` or `rate_limited`), and never anything
derived from the password. The file can rotate by size
(`auth.usage.log_max_bytes`, `auth.usage.log_max_backups`); with an external
logrotate, send `SIGHUP` to reopen it. To add up successful audio per user:

```bash
jq -s 'map(select(.status == "ok")) | group_by(.user)
       | map({user: .[0].user, audio_seconds: (map(.audio_seconds) | add)})' requests.jsonl
```

**The totals** are one entry per user:

| Field | Meaning |
|---|---|
| `requests`, `ok_requests`, `failed_requests` | number of jobs |
| `audio_seconds` | audio of successful jobs: the amount transcribed for this user |
| `failed_seconds` | audio of jobs that failed |
| `unknown_requests` | failed jobs whose audio length was not known |
| `processing_seconds` | time the engines spent on this user's jobs (elapsed minus queued) |
| `first_seen`, `last_seen` | time of the first and the latest job |
| `by_protocol`, `by_model` | `requests`, `ok_requests`, `failed_requests` and `audio_seconds` per protocol and per model |

The totals are always kept in memory, counting from process start. With a
totals file they are loaded at startup, written every
`auth.usage.snapshot_interval` (30s) when something changed, and written
once more at shutdown; missing parent directories are created. Which file:

- `auth.usage.totals_file`, when set in the YAML file or through
  `WHISPER_HUB_AUTH_USAGE_TOTALS_FILE`. If its directory cannot be
  created, startup fails.
- Otherwise `auth.usage.default_totals_file`, which the images set to
  `/data/usage.json` through `WHISPER_HUB_AUTH_USAGE_DEFAULT_TOTALS_FILE`.
  If `/data` is not writable (a read-only mount, a `--user` without write
  access), the gateway logs one warning and keeps the totals in memory.
- Otherwise none: the bare binary keeps the totals in memory.

To keep the totals in memory in an image, set
`WHISPER_HUB_AUTH_USAGE_DEFAULT_TOTALS_FILE=` (empty). It has to be the
variable: the image sets it in the environment, which overrides the YAML
file.

When `auth.usage.totals_file` is set but the file is missing or empty, and
the default file has content, the totals are moved: the new file is written
first, then the old one is removed, and one `usage totals moved` line at
Info names both. A totals file that already has content is used as it is,
and the old file is left alone. If the move fails, the gateway warns, starts
anyway and keeps the old file.

The file is a JSON object with the user names as keys:

```bash
jq 'map_values(.audio_seconds / 3600)' usage.json    # hours of audio per user
```

**The admin endpoint** returns the same totals, current to the second, as
`{"users": {"<name>": {...}}}`. It is served to admin access: every
request while authentication is off, and every request on the admin port
`http.open_listen`. On the authenticated listener it needs the credentials
of a user with `admin: true`, whatever `auth.protected_paths` says; other
users get 403. `?user=<name>` narrows the answer to one user:

```bash
curl -H "Authorization: Bearer $ADMIN_KEY" "http://localhost:9000/_hub/usage?user=alice"
```

**With authentication off**, accounting still works: all HTTP jobs are
recorded for the user `admin`, and Wyoming jobs, as always, for the user
`wyoming`. Jobs without credentials
while authentication is on (unprotected paths, `http.open_listen`) are
recorded for `admin` too, in the same entry as a users file account named
`admin` if there is one. The log line, the request log file, the totals
file and `/_hub/usage` behave as described.

## Migrating from whisper-asr-webservice

Replace the image; keep the environment, the flags, the port and the cache
volume. whisper-hub's own names all start with `WHISPER_HUB_`; the names an
upstream deployment already uses are accepted as compatibility aliases, so
it keeps working unchanged. This table is the complete list. When both the
`WHISPER_HUB_` name and its alias are set, the `WHISPER_HUB_` name wins; an
empty value counts as unset for the worker's settings.

| Upstream name (alias) | whisper-hub name | Meaning |
|---|---|---|
| `ASR_ENGINE` | `WHISPER_HUB_ENGINE` | the engine. On `whisper-hub:all` exactly as upstream (unset = `openai_whisper`); on a single-engine image it must match the image (unset = the image's engine) |
| `ASR_MODEL` | `WHISPER_HUB_ENGINE_MODEL` | the model, same meaning |
| `ASR_MODEL_PATH` | `WHISPER_HUB_ENGINE_MODEL_PATH` | where models are downloaded to |
| `ASR_DEVICE` | `WHISPER_HUB_ENGINE_DEVICE` | `cuda` or `cpu` |
| `ASR_QUANTIZATION` | `WHISPER_HUB_ENGINE_QUANTIZATION` | the compute type |
| `MODEL_IDLE_TIMEOUT` | `WHISPER_HUB_ENGINE_MODEL_IDLE_TIMEOUT` | seconds before an unused model is unloaded |
| `SUBTITLE_MAX_LINE_WIDTH`, `SUBTITLE_MAX_LINE_COUNT`, `SUBTITLE_HIGHLIGHT_WORDS` | `WHISPER_HUB_ENGINE_SUBTITLE_MAX_LINE_WIDTH`, `..._MAX_LINE_COUNT`, `..._HIGHLIGHT_WORDS` | whisperx subtitle output |
| `LOG_LEVEL` | `WHISPER_HUB_LOG_LEVEL` (`log.level`) | both processes; each accepts `debug`, `info`, `warn` or `warning`, `error`, `critical`, in any case |
| `UPLOAD_SPOOL_DIR` | `WHISPER_HUB_AUDIO_SPOOL_DIR` (`audio.spool_dir`) | the gateway's spool directory for uploads |
| `DEBUG_FAILED_UPLOADS_DIR` | `WHISPER_HUB_AUDIO_FAILED_UPLOADS_DIR` (`audio.failed_uploads_dir`) | dumps of uploads that fail to decode |
| `SAMPLE_RATE` | `WHISPER_HUB_LEGACY_RAW_SAMPLE_RATE` (`legacy.raw_sample_rate`) | the sample rate of `encode=false` uploads |
| `--host`/`-h`, `--port`/`-p` | the same flags | the HTTP listener. The container command may still be `whisper-asr-webservice ...` |
| `/root/.cache` volume | the same path | reused |

`HF_TOKEN` keeps its name: it belongs to Hugging Face. `LOG_LEVEL`,
`UPLOAD_SPOOL_DIR` and `DEBUG_FAILED_UPLOADS_DIR` are not variables of
upstream 1.10.0; they come from a patched upstream build and are accepted
for deployments that used it.

## Compatibility

### What is promised

- The legacy API stays compatible with upstream whisper-asr-webservice for
  whisper-hub major versions 0 and 1. From 2.0 on this compatibility may
  be dropped.
- Compatible means API compatibility: the same paths, parameters, status
  codes and response shapes, with the [extensions](#extensions-to-upstream)
  listed below. whisper-hub does not promise byte-for-byte identity with
  upstream. The byte-level comparison described below is a verification
  tool for the project, not part of the promise.

### The reference

whisper-hub is compared with a **reference**: public upstream
[whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice),
`main` at the commit pinned in
[`deploy/e2e/reference/upstream.env`](deploy/e2e/reference/upstream.env)
(release 1.10.0), built from that commit and run unmodified:

```bash
make image-reference   # docker build -t whisper-asr-webservice:ec32cd5 \
                       #   "https://github.com/ahmetoner/whisper-asr-webservice.git#<commit>"
```

The worker's engine libraries (`worker/requirements/constraints.txt`) are
exactly the versions that image installs. The comparison sends the same
upload to both and checks status code, headers and body, allowing for
whisper-hub's extensions below and the differences listed after them.

### Extensions to upstream

whisper-hub deliberately answers these differently from upstream. The
recorded-case replay (`internal/legacy/reference_test.go`) and the fidelity
comparison (`deploy/e2e/fidelity.py`) apply exactly these on top of the
reference's responses and report anything else as a difference:

1. `/asr` and `/detect-language` report the length of the decoded audio in
   an `X-Audio-Duration-Seconds` header (also on a failed transcription);
   `/detect-language` also in the field `audio_duration_seconds`.
2. An engine failure gets `500 {"detail": "Transcription failed"}` (or
   `"Language detection failed"`); upstream answers a plain-text
   `Internal Server Error`.
3. An upload that cannot be decoded, or raw PCM of odd length, gets
   `400 {"detail": "<reason> (<size> bytes, magic=<first bytes in hex>)"}`;
   upstream answers a plain-text 500.
4. An upload that decodes to no audio at all gets the same kind of 400
   (`Audio decoded to 0 samples ...`); upstream answers 200 with an empty
   transcript.
5. Uploads are written to disk and ffmpeg reads them from the file;
   upstream pipes them into ffmpeg. MP4 and M4A recordings with the index
   at the end of the file (as Android writes them) therefore decode, where
   upstream gets no audio; and MP3 encoder delay is trimmed (it is only
   readable from a seekable file), so MP3 transcripts can differ slightly
   from upstream's. WAV, Ogg and the other formats decode to the same
   samples either way.
6. Settings upstream 1.10.0 does not have: the log level, the spool
   directory and the dumps of failed uploads (the aliases of a patched
   upstream build are in the [drop-in table](#migrating-from-whisper-asr-webservice)),
   and Hugging Face URLs as the model (faster_whisper and whisperx).

### Known differences

Legacy API, besides the extensions above:

- The gateway is ready before the model: it answers 503 with
  `{"detail": ...}` while the worker loads, where upstream does not listen
  yet. `/healthz` is new.
- There is no `Server: uvicorn` header, and header names are in canonical
  case.
- On the single-engine images, an unset `WHISPER_HUB_ENGINE` means the image's
  engine, not `openai_whisper` (`whisper-hub:all` behaves as upstream).
- whisperx with `initial_prompt` works; upstream answers 500, because its
  whisperx version rejects the argument.
- whisperx starts at all: the worker allowlists the classes of the voice
  activity model bundled with whisperx for torch's safe loading. Upstream's
  image fails to start whisperx for that reason (the fidelity script gives
  the reference the same allowlist, `deploy/e2e/reference/sitecustomize.py`,
  and nothing else).
- Two upstream oddities are kept for fidelity. whisperx diarization passes
  `min_speakers` and `max_speakers` by position, as upstream does, so
  `min_speakers` is taken as the exact number of speakers and `max_speakers`
  as the minimum. And the
  `/detect-language` confidence of openai_whisper is the probability of
  `zh`, not of the detected language, as upstream computes it.
- `Asr-Engine` names the engine that ran the job, since several workers may
  have different engines.
- Additions: the `model` query parameter; 413 when an upload exceeds
  `legacy.max_upload_bytes`; 401, 403, 404 and 503 for authentication,
  model access, unknown models and unavailable workers.
- HTTP edge cases follow Go's `net/http` rather than uvicorn: a path like
  `//asr` gets a 301 instead of a 404, the order of the `Allow` header is
  fixed (upstream's varies per process), and request limits and the
  handling of malformed HTTP differ.
  Speaker counts beyond the integer range saturate. Multipart charsets
  other than utf-8 and latin-1 are read as utf-8.

OpenAI API, compared with `whisper-1`:

- There is no 25 MB upload limit by default (`openai.max_upload_bytes`).
- Unknown model names fall back to the default model.
- `response_format=diarized_json` is rejected, since Whisper models have no
  speaker diarization here. Fields that only OpenAI's `gpt-*` models use
  (`chunking_strategy`, `include`) are accepted and ignored.
- `usage` reports whole seconds, rounded up; `owned_by` is `whisper-hub`.
- An engine failure in the middle of a stream is reported with the
  gateway's own message, not the engine's.

Wyoming:

- Model ids are listed, aliases are not. The program name is `whisper-hub`.
- `zh-HK` is advertised for Cantonese (`yue`), as wyoming-faster-whisper
  does, because Home Assistant cannot select `yue` otherwise.
- Transcript streaming events are off by default
  (`wyoming.stream_transcripts`): Home Assistant reads only the final
  transcript.

## Operations

- **Logs** go to stderr: one Info line per job (see
  [Usage accounting](#usage-accounting)), and every step at Debug
  (`log.level: debug` or `WHISPER_HUB_LOG_LEVEL=debug`). In the all-in-one image the
  worker writes its own lines into the same stream.
- **`SIGTERM`, `SIGINT`**: stop accepting, let running requests finish for
  `http.shutdown_grace` (30s), then cut them off (which also stops their
  worker jobs), stop the managed worker, and write the final usage totals.
- **`SIGHUP`**: reload the users file and reopen the request log.
- **Health check**: `GET /healthz`, or `whisper-hub healthcheck` inside the
  container.

## Development

```bash
make check    # gofmt, go vet, go test -race ./...
make image    # build the all-in-one image locally
```

- [AGENTS.md](AGENTS.md): how to build, test and verify, the repository
  layout, the conventions, and how to make common changes.
- [docs/DESIGN.md](docs/DESIGN.md): the architecture and the reasons for
  it.
- [docs/worker-api.md](docs/worker-api.md): the internal HTTP API between
  gateway and worker.
- [worker/README.md](worker/README.md): the engine worker and its settings.
