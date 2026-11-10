# whisper-hub design

This document describes how whisper-hub is built and why. It is for people
who want to understand or change the code. How to run the service is in the
[README](../README.md); the working rules for contributors and coding agents
are in [AGENTS.md](../AGENTS.md).

## Goals and non-goals

whisper-hub is one speech-to-text service that speaks three client protocols
at once:

1. **Legacy**: the HTTP API of
   [whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice)
   (`/asr`, `/detect-language`, `/docs`). The goal is full compatibility:
   an existing deployment (same port, same environment variables, same
   flags, same model cache) and its clients keep working when the image is
   swapped, and responses are identical byte for byte.
2. **OpenAI**: `/v1/audio/transcriptions`, `/v1/audio/translations` and
   `/v1/models`, compatible at the level of OpenAI's `whisper-1` model, so
   that official SDKs work unchanged.
3. **Wyoming**: the TCP protocol Home Assistant uses for voice pipelines.

On top of that it provides authentication, per-user model access and usage
accounting, which whisper-asr-webservice does not have.

Non-goals:

- Parity with OpenAI's `gpt-*` transcription models (diarized output, log
  probabilities, server-side chunking) and the OpenAI Realtime API.
- Decoding or resampling audio in the workers.
- Preempting a running job, a persistent job queue, or sharing state between
  several gateway processes.
- Platforms other than linux/amd64 containers.

## Terms

| Term | Meaning |
|---|---|
| upstream | whisper-asr-webservice, the service whose API the legacy protocol reproduces |
| reference | public upstream at a pinned commit, built and run unmodified, that whisper-hub is compared with byte for byte, see [Legacy compatibility](#legacy-compatibility) |
| engine | a speech recognition library: `faster_whisper`, `whisperx`, `openai_whisper`, or the model-free `fake` used in tests |
| worker | a Python process that runs one engine and serves one or more models over the internal [worker API](worker-api.md) |
| gateway | the Go binary `whisper-hub`: everything except running models |
| frontend | one of the three protocol implementations inside the gateway |
| hub | the part of the gateway behind the frontends: worker registry, model resolution, scheduling |
| job | one transcription or language detection handed to the hub |
| model id | the name a worker reports for a model, for example `base` or `large-v3` |
| alias | an extra public name for a model id, for example `whisper-1` |

## Architecture

```
clients --> gateway (Go, one static binary)            --> engine worker(s) (Python)
            :9000  HTTP  legacy + OpenAI + /healthz         :9100 internal HTTP
            :10300 TCP   Wyoming                            faster_whisper | whisperx |
            auth, model access, usage accounting            openai_whisper | fake
            ffmpeg decode -> PCM s16le 16 kHz mono
            one priority queue for all workers
```

The system is split into two kinds of process, and the split follows what
each language is needed for:

- **The gateway owns everything protocol-shaped.** Parsing three protocols,
  authentication, access rules, queueing and accounting are plain network
  code with no dependency on a machine learning library. In Go this is one
  static binary of a few megabytes that starts instantly.
- **Workers only run models.** The engines are Python libraries with heavy
  dependencies (CTranslate2, torch). A worker is a thin HTTP shell around
  one engine and knows nothing about clients, users or protocols.

Consequences of the split:

- **Audio is decoded once, in the gateway.** Every frontend turns its input
  into the same thing: mono 16 kHz signed 16-bit PCM. Workers receive that
  PCM and need no ffmpeg. The gateway also knows the audio duration before
  any model runs, which is what usage accounting needs.
- **Several workers can be attached.** A worker serves one engine and one or
  more models. Different engines, or several copies of one model, are just
  more entries in `hub.workers`.
- **Adding a protocol or an engine touches one side only.** A new frontend
  is written against the Go contracts in `internal/core`; a new engine is a
  Python class behind the worker API.
- **The cost is a second process and an internal API** to keep in sync on
  both sides. The all-in-one image hides the second process from users, see
  [Images and process supervision](#images-and-process-supervision).

## Modules and their interfaces

### Gateway packages

| Package | What it is | Interface (what callers use) |
|---|---|---|
| `internal/core` | the contracts every other package shares | types `Request`, `Result`, `Segment`, `Model`, `Identity`, `Error`, `UsageEvent`; interfaces `Service`, `AudioDecoder`, `Authorizer`, `UsageRecorder` |
| `internal/legacy` | legacy frontend | `New(cfg, service, decoder)`, `Register(mux)` |
| `internal/openaiapi` | OpenAI frontend | `New(cfg, service, decoder, log)`, `Register(mux)`, `WriteError` (so the auth middleware can answer in OpenAI's error shape) |
| `internal/wyoming` | Wyoming frontend: codec, TCP server, zeroconf announcement | `New(cfg, service, decoder, identify, log)`, `Serve(ctx, listener)` |
| `internal/auth` | users file, HTTP authentication, model access rules, usage accounting | `New(cfg, rejecters)`, `Middleware`, `WyomingIdentity`, `UsageHandler`, `Reload`, `Run`, `Close`; it is the `core.Authorizer` and `core.UsageRecorder` |
| `internal/hub` | the `core.Service` implementation: worker health, model resolution, scheduling | `New(cfg, authorizer, recorder, log)`, the `core.Service` methods, `HealthHandler`, `Close` |
| `internal/hub/internal/workerclient` | HTTP client of the worker API, private to the hub | used by `internal/hub` only |
| `internal/hub/workertest` | an in-process fake worker for tests | `New`, scripted responses, recorded jobs |
| `internal/audio` | the `core.AudioDecoder` implementation: ffmpeg for files, pure Go conversion for raw PCM | `New(cfg, log)`, `Check`, `DecodeFile`, `FromRaw` |
| `internal/config` | the whole configuration: defaults, YAML file, environment, flags, upstream variables | `Load(args, getenv, out)`, `Config` |
| `internal/workerproc` | the worker as a child process of the gateway | `Start(cfg, env, log)`, `Proc.Done`, `Proc.Stop`, `Config.Env` |
| `cmd/whisper-hub` | the binary: builds the modules, wires them together, handles signals | the command line |

### The core contracts

`core.Service` is the single entry point a frontend uses:

```go
type Service interface {
    Transcribe(ctx context.Context, req *Request, onSegment func(Segment) error) (*Result, error)
    DetectLanguage(ctx context.Context, req *DetectRequest) (*Detection, error)
    Models(ctx context.Context) []Model
}
```

Its interface is more than these signatures. A frontend may rely on the
following, and must not repeat any of it:

- The caller's identity travels in `ctx` (`core.IdentityFrom`). A context
  without one is the anonymous caller.
- `Request.Model` is passed exactly as the client sent it. The service
  resolves aliases, applies the unknown-model policy and enforces access.
- The service queues the job by `Request.Priority` and records usage, for
  failed jobs too.
- `onSegment`, when given, is called for each segment as soon as the engine
  produces it. An error returned from it cancels the job.
- Canceling `ctx` cancels the job, whether it is queued or running.
- Every error is a `*core.Error` with a kind (`bad_request`, `bad_audio`,
  `unauthenticated`, `forbidden`, `model_not_found`, `unavailable`,
  `canceled`, `internal`) and a message that is safe to show to a client.
  The underlying cause is for the log only.
- `Models` lists only what the caller may use, default model first.

`core.AudioDecoder` turns client audio into PCM: `DecodeFile` for anything
ffmpeg understands, `FromRaw` for headerless PCM of any rate, sample width
and channel count.

`core.Authorizer` (`AllowModel`) and `core.UsageRecorder` (`Record`) are what
the hub needs from authentication. Both are implemented by `auth.Auth`; the
hub accepts `nil` for either.

### Dependency rules

- A frontend imports `internal/core` and nothing else from this repository.
  Frontends do not know about each other, the hub, authentication or
  configuration files.
- `internal/hub`, `internal/auth` and `internal/audio` also import only
  `internal/core`.
- Each package exports a `Config` struct with `yaml` tags whose zero values
  mean "the default", and a constructor that takes this config plus the core
  interfaces it needs. No package reads environment variables or
  configuration files itself.
- Only `internal/config` (which assembles the packages' `Config` structs
  into one) and `cmd/whisper-hub` (which constructs and connects the
  modules) import the other packages. `main` composes and holds no logic.
- HTTP frontends do not authenticate. They read the caller that
  `auth.Middleware` put into the request context.

### The worker

The worker (`worker/whisper_worker/`) mirrors the same idea in Python:

| Module | What it is |
|---|---|
| `app.py` | the worker API as a plain ASGI application; watches for client disconnects while streaming |
| `config.py` | configuration from environment variables; upstream's variables keep upstream's meaning |
| `models.py` | model lifecycle (lazy loading, eviction, idle unloading) and one job at a time per model |
| `engines/base.py` | the `Engine` interface (`load`, `unload`, `transcribe`, `detect_language`, `capabilities`) and the engine-neutral segment shape |
| `engines/*.py` | one adapter per engine |
| `legacy_writers.py` | the output writers upstream uses for faster_whisper, copied so that output is identical |
| `model_resolver.py` | Hugging Face URLs as model names |

### Worker API

The interface between gateway and worker is a small internal HTTP API,
specified in full in [worker-api.md](worker-api.md). Its shape follows from
what the gateway needs:

- A job is one multipart request: a JSON part with engine-neutral options
  and a part with the raw PCM. The worker never sees a client's file.
- The transcript comes back as a stream of JSON lines, one per segment as
  the engine produces it, so that streaming clients get text early and the
  gateway can tell a working engine from a stuck one.
- The gateway cancels a job by closing the connection. No job ids or cancel
  calls are needed.
- A worker describes itself (`GET /v1/info`: engine, models, optional
  features), so the gateway needs no per-worker configuration beyond a URL.
- The API is unauthenticated and unversioned: gateway and worker are
  released together, and the worker port is never exposed to clients.

## Request flow

All three flows meet in `core.Service.Transcribe`. What differs is how the
request is read and how the result is written.

### Legacy: `POST /asr`

1. `auth.Middleware` checks the credentials when authentication is on and
   stores the identity in the request context.
2. `internal/legacy` reads the query string the way FastAPI does and the
   body the way Starlette does (see
   [Legacy compatibility](#legacy-compatibility)). The `audio_file` part is
   streamed straight into the decoder: with `encode=true` (the default) into
   ffmpeg, with `encode=false` it is taken as raw PCM at
   `legacy.raw_sample_rate`.
3. It calls `Transcribe` with `LegacyOutput` set to the `output` parameter,
   priority normal and no segment callback.
4. The hub resolves the model, checks access, drops options the engine does
   not support (upstream ignores them silently), queues the job and sends it
   to a worker with `legacy_output` set.
5. The worker returns the transcript already rendered in the requested
   format (`Result.Rendered`). The frontend writes these bytes unchanged,
   with upstream's headers: `Asr-Engine`, `Content-Disposition`,
   `Content-Type: text/plain; charset=utf-8`, plus whisper-hub's own
   `X-Audio-Duration-Seconds`.

`POST /detect-language` is the same with `DetectLanguage`; the JSON answer
is built in the frontend. `/`, `/docs`, `/redoc`, `/openapi.json` and
`/assets/*` are served from files embedded in the binary.

### OpenAI: `POST /v1/audio/transcriptions`

1. `auth.Middleware`, as above. Rejections under `/v1/` are written by
   `openaiapi.WriteError`, so SDKs get the error objects they expect.
2. `internal/openaiapi` reads the multipart form field by field, validates
   each field on arrival with OpenAI's error codes, and streams the `file`
   part into ffmpeg.
3. It calls `Transcribe` with priority normal. Word timestamps are requested
   from the engine only when the response can carry them (`verbose_json`
   with `timestamp_granularities` containing `word`).
4. The response is rendered in Go from the engine-neutral segments: `json`,
   `text`, `srt`, `vtt` or `verbose_json`.
5. With `stream=true` and format `json` or `text`, the frontend passes a
   segment callback and sends each segment as a server-sent event
   `transcript.text.delta`, then `transcript.text.done`. The response status
   is sent with the first event, so a failure before any segment is still a
   normal HTTP error.

`POST /v1/audio/translations` sets the task to translate and ignores
`language`, `stream` and `timestamp_granularities`, as `whisper-1` does.
`GET /v1/models` lists model ids and aliases the caller may use.

### Wyoming

1. On connect, `auth.WyomingIdentity` maps the remote address to an identity
   (Wyoming has no credentials). A refused connection gets an `error` event.
2. `describe` is answered with `info`: one speech-to-text program whose
   models are the ones the identity may use.
3. `transcribe` names the model and language for the next utterance.
   `audio-start`, `audio-chunk` and `audio-stop` carry the audio, which is
   buffered up to `wyoming.max_utterance_bytes` and
   `wyoming.max_utterance_duration`.
4. At `audio-stop` the buffered audio is converted with
   `AudioDecoder.FromRaw` and passed to `Transcribe` with priority
   interactive. The requested language is mapped to a Whisper code the model
   supports (`zh-HK` to `yue`, `pt-BR` to `pt`); anything unusable means
   automatic detection.
5. The answer is a `transcript` event. With `wyoming.stream_transcripts` it
   is wrapped in `transcript-start`, `transcript-chunk` and
   `transcript-stop`.
6. A failed job sends an `error` event and closes the connection, because
   older Home Assistant versions notice a failure only that way.

A connection runs one transcription at a time; up to two more finished
utterances may wait behind it.

### Errors per protocol

The hub reports failures as error kinds; each frontend translates them:

| Kind | Legacy | OpenAI | Wyoming |
|---|---|---|---|
| `bad_request` | 500 `{"detail":"Transcription failed"}` (every engine exception is a 500, as upstream; the JSON detail is an extension) | 400 `invalid_request_error` | `error` event |
| `bad_audio` | 400 with a redacted detail (an extension; upstream answers 500) | 400, `param` `file`, `code` `invalid_value` | `error` event |
| `unauthenticated` | 401 with a Basic challenge | 401 `invalid_api_key` | `error` event at connect |
| `forbidden` | 403 `{"detail": ...}` | 403 `model_not_allowed` | `error` event |
| `model_not_found` | 404 `{"detail": ...}` | 404 `model_not_found` | `error` event |
| `unavailable` | 503 `{"detail": ...}` | 503 `server_error` | `error` event |
| `internal` | 500 `{"detail":"Transcription failed"}` | 500 `server_error` | `error` event |
| `canceled` | nothing: the client is gone | 499, for the log | nothing |

Invalid legacy query parameters never reach the hub: the frontend answers
them with FastAPI's 422 body.

## Model selection and access

Workers report their models. The hub merges the reports of all healthy
workers into one list of model ids and adds the configured aliases.

A model name from a client is resolved in this order:

1. Empty: the default model. That is `hub.default_model`, or, when it is not
   set, the default model of the first healthy worker in configuration order.
2. The id of a model that is being served: that model.
3. An alias from `hub.aliases`: its target; an empty target means the
   default model. `whisper-1` is always an alias of the default model unless
   it is mapped to something else.
4. The id of a model that was served earlier but has no healthy worker now:
   the job fails as `unavailable`. It never falls back to another model.
5. Anything else is an unknown name. `hub.unknown_model: default` (the
   default) uses the default model, because many OpenAI clients hardcode
   model names; `error` fails the job as `model_not_found`.

After resolution the hub asks the `Authorizer` whether the caller may use
the model:

- A user may always use the default model, plus every model id matched by
  one of the `models` patterns of the user's entry. A pattern is an exact id
  or a glob in which `*` matches any run of characters, `/` included.
- A request for a model the user may not use fails as `forbidden`. It never
  silently falls back to the default model.
- Anonymous callers (authentication off) may use every model, unless
  `auth.anonymous.restrict` limits them to the default model plus
  `auth.anonymous.models`.

`Models` applies the same rule, which is why `/v1/models` and the Wyoming
`info` answer show only what the caller may use.

When several workers serve one model id, the model has an optional feature
only if all of them have it, so that a job is valid on whichever worker it
lands.

Options the engine cannot honor are handled per protocol. Legacy requests
never fail on them, because upstream ignores them: the option is dropped,
and `translate` becomes `transcribe`. Other protocols get `bad_request` for
options that would change the shape of the answer (translation, word
timestamps, diarization). `vad_filter` is only a hint and is dropped for
everyone.

## Authentication

Authentication is off until `auth.users_file` is set. While it is off, every
caller is anonymous, as with upstream.

When it is on:

- `auth.Middleware` wraps the HTTP mux. Paths under `auth.protected_paths`
  (default `/asr`, `/detect-language`, `/v1/`) need credentials; everything
  else is public. `OPTIONS` requests pass without credentials, because
  browsers send CORS preflights without them. The path is cleaned before
  matching, so `/v1/../asr` cannot dodge the rule.
- Accepted credentials: HTTP Basic with a bcrypt password hash,
  `Authorization: Bearer <api key>`, and `X-Api-Key: <api key>`.
- The users file holds hashes only, so it can be reviewed and mounted
  read-only. Passwords use bcrypt at cost 12, because an offline attack on a
  leaked file is the realistic threat. API keys are 256 random bits and are
  stored as SHA-256: a slow hash would add nothing against guessing and
  would cost every request.
- bcrypt takes about 250 ms by design. A verified Basic login is therefore
  remembered for 5 minutes, keyed by an HMAC under a per-process secret over
  the name, password and stored hash. A reload of the users file clears this
  cache, so a removed user stops working at once.
- Failed attempts are counted per client address in fixed windows
  (`auth.rate_limit`). A blocked address is rejected without a bcrypt
  evaluation; cached logins and valid API keys still pass, since they cost
  no bcrypt time. Unknown users take as long as known ones.
- The users file is polled for changes (`auth.poll_interval`) rather than
  watched: editing the file in a running container should be enough, and
  inotify is unreliable across bind mounts. A file that does not parse keeps
  the previous users. No users file problem prevents startup: a gateway
  that will not start cannot be fixed from inside its container.
- Each API rejects in its own dialect. Legacy paths get a plain-text 401
  with a Basic challenge, so browsers prompt for credentials; `/v1/` gets
  an OpenAI error object.
- Wyoming carries no credentials. Connections act as the users file entry
  named in `auth.wyoming.user`, optionally restricted to source networks in
  `auth.wyoming.allow_cidrs`.
- `http.open_listen` serves the same routes on a second address without the
  middleware, for clients whose access the network already restricts.

## Usage accounting

The hub creates one usage event per job, when the job finishes, whatever the
outcome: user, protocol, endpoint, resolved model, engine, task, seconds of
audio, time spent queued, total time, status and error text. The audio
length is known before the job runs because the gateway decoded the audio.
The event is recorded with a context detached from the request, so a job
whose client disconnected is still recorded.

`auth` receives the event and does three things with it:

1. Updates the per-user totals in memory. Seconds of successfully
   transcribed audio and seconds of audio whose job failed are kept apart:
   adding them would either overcharge for failed work or hide how much
   processing is being lost.
2. Appends one JSON line to the request log, if `auth.usage.log_file` is
   set. The write is synchronous: one small write per job is cheap next to a
   transcription.
3. Writes the one Info log line of the job.

The totals are written to `auth.usage.totals_file` (while it is unset, to
`auth.usage.default_totals_file`, which the images point at
`/data/usage.json`) atomically (temporary file, fsync, rename) every
`auth.usage.snapshot_interval` when they changed, and once more at
shutdown. They are served to admin users at
`GET /_hub/usage`.

Requests that never become a job are not usage: a failed login, an invalid
parameter, an upload that does not decode. Failed logins are written to the
request log for auditing, without any derivative of the password, and are
deliberately kept out of the totals so that nobody can create entries for
names of their choosing.

Accounting works with authentication off; the user is then `anonymous`.

The field lists and the operator's view are in the
[README](../README.md#usage-accounting).

## Scheduling and priorities

Each worker runs at most `max_in_flight` jobs (default 1). Jobs that cannot
start wait in one queue for the whole hub, bounded by `hub.queue_length`.

- Whenever a slot frees up or a worker appears, waiting jobs are considered
  by effective priority, highest first, then by arrival. Each goes to the
  least loaded healthy worker that serves its model and has a free slot;
  equally loaded workers take turns.
- A job whose model has no free worker does not block jobs for other models.
- There are three priorities: batch, normal and interactive. Legacy and
  OpenAI requests are normal; Wyoming requests are interactive, so a
  two-second voice command waits only for the job already running, not
  behind an hour-long file. No frontend submits batch jobs.
- The effective priority is the requested one plus one level for every
  `hub.priority_aging` interval the job has waited, capped at interactive.
  This bounds starvation: after one interval a normal job competes with
  interactive ones in arrival order. A negative interval disables aging.
- Running jobs are never preempted.
- When the queue is full, a new job pushes out the newest waiting job of
  strictly lower effective priority, which fails as `unavailable`. If there
  is none, the new job itself fails as `unavailable`.
- A waiting job whose model loses its last healthy worker fails at once
  instead of waiting for an unknown time.

Worker health is tracked by probing: `GET /healthz`, then `GET /v1/info`.
A healthy worker is probed every `hub.poll_interval`. One that is down or
still loading is probed with exponential backoff between `hub.retry_min` and
`hub.retry_max`. A healthy worker leaves rotation after `hub.fail_threshold`
consecutive failed probes, because a busy Python process can miss one. A job
that fails with a transport error triggers an immediate probe. The gateway
never waits for workers at startup; until the default model is served by a
healthy worker, `/healthz` answers 503 and jobs fail as `unavailable`.

## Legacy compatibility

### What it is measured against

The reference is public upstream
[whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice),
`main` at the commit in `deploy/e2e/reference/upstream.env` (release
1.10.0). `make image-reference` builds it from that commit with upstream's
own Dockerfile, and it runs as built, with its own `app/`. Anyone can
rebuild it, and the image installs exactly the library versions the worker
pins.

whisper-hub reproduces it, plus a short list of its own extensions: the
audio duration in a header and in the `/detect-language` JSON, JSON details
for engine failures, a redacted 400 for audio that does not decode or has no
samples, decoding from a spool file instead of a pipe (MP4 with the index at
the end decodes, MP3 encoder delay is trimmed), and the
`LOG_LEVEL`, `UPLOAD_SPOOL_DIR` and `DEBUG_FAILED_UPLOADS_DIR` variables. The
list, with what upstream answers instead, is in the
[README](../README.md#extensions-to-upstream); the guards below apply
exactly these extensions to upstream's responses, and nothing else.

### How identical output is achieved

A transcript is the product of four things: the decoded audio, the model and
library that process it, the code that formats the result, and the HTTP
framework around it. Each is matched separately.

1. **Same audio.** The gateway images ship the exact ffmpeg build of the
   upstream image (`onerahmet/ffmpeg:n7.1`) and call it for the same output
   format, so an upload decodes to the same samples, except where reading
   from a file instead of a pipe changes them (MP4 with a trailing index,
   MP3; extension 5 in the README).
2. **Same engine code path.** Each worker engine makes the same library
   calls with the same arguments as the corresponding upstream engine
   module, and the Python package versions are pinned to those of the
   reference image (`worker/requirements/constraints.txt`, the reference
   image's `pip freeze`).
3. **Same formatting code.** The response body is rendered inside the
   worker: by the library's own writers for openai_whisper and whisperx, and
   by a copy of upstream's writer classes for faster_whisper. The gateway
   passes the bytes through and never reformats a legacy transcript.
4. **Same HTTP behavior.** `internal/legacy` reproduces what FastAPI,
   Starlette, python-multipart and the Python standard library do around
   the body:
   - query parsing: the last value of a repeated parameter wins, booleans
     and integers accept what pydantic accepts, and validation errors are
     FastAPI's 422 JSON in declaration order;
   - multipart parsing: a port of python-multipart, because Go's
     `mime/multipart` differs in observable ways (it accepts a preamble,
     decodes RFC 2231 filenames, strips directories from filenames);
   - Python's JSON string escaping and float formatting, latin-1 header
     encoding, and Starlette's redirects for trailing slashes;
   - the Swagger and ReDoc pages, the bundled assets with their `ETag` and
     `Last-Modified` values, and one OpenAPI schema per engine, all captured
     from the reference and embedded in the binary.

Two upstream quirks are kept on purpose, because removing them would change
output: whisperx diarization passes `min_speakers` and `max_speakers`
positionally, so they land in the wrong parameters of the diarization
pipeline; and the `/detect-language` confidence of openai_whisper is the
probability of `zh`, not of the detected language.

### How it is guarded

- `internal/legacy/testdata/reference/cases/` holds recorded exchanges with
  the reference: ordinary requests, every validation error, malformed and
  truncated bodies, odd filenames, static pages. `go test` replays each
  request against the frontend, with a fake service that returns the
  reference's own transcript bytes, and compares status, headers and body
  with the recorded response plus the extensions (`withExtensions` in
  `reference_test.go`).
- The worker's tests pin the copied writer code and compare it with the
  installed faster-whisper library.
- An end-to-end comparison sends the same uploads to a running reference and
  to a whisper-hub image and compares every response, again with the
  extensions applied explicitly (`deploy/e2e/fidelity.py`). Its results are in the
  [README](../README.md#verification-status); how to run it is in
  [AGENTS.md](../AGENTS.md#end-to-end-checks).

The deliberate deviations are listed in the
[README](../README.md#known-differences).

## Configuration

All configuration is resolved in one place, `internal/config`, in increasing
precedence: built-in defaults, one YAML file, the variables of an upstream
deployment, `WHISPER_HUB_*` variables, command line flags.

- The environment variable of a setting is derived from its YAML path by
  reflection, so a new field gets its variable without extra code.
- Unknown keys in the YAML file, and in the users file, are errors. A
  misspelled key would otherwise be ignored silently.
- Zero values mean "default" and each package resolves its own defaults, so
  the defaults live next to the code that uses them.
- Upstream's variables and flags (`--host`, `--port`, `LOG_LEVEL`,
  `SAMPLE_RATE`, ...) are mapped onto the same settings, which is what
  makes the image a drop-in replacement.
- Worker settings are not part of this: the worker reads its own environment
  variables, and in the all-in-one image it inherits the container's
  environment unchanged.

## Images and process supervision

| Image | Built from | Contents |
|---|---|---|
| all-in-one, one engine | `deploy/Dockerfile`, `ENGINE=<engine>` | gateway, ffmpeg, one engine's worker |
| all-in-one, all engines | `deploy/Dockerfile`, `ENGINE=all` | gateway, ffmpeg, the three real engines in one Python environment |
| gateway | `deploy/Dockerfile.gateway` | gateway and ffmpeg on a minimal base, running as an unprivileged user |
| worker | `worker/Dockerfile`, `ENGINE=<engine>` | one engine's worker |
| CUDA variants | `deploy/Dockerfile.gpu`, `worker/Dockerfile.gpu` | the same on an NVIDIA CUDA runtime base |

In the all-in-one image the gateway is PID 1 and starts the worker as its
child (`internal/workerproc`, enabled by `worker.command`). There is no
supervisor script, for three reasons:

- Signals reach the process that knows the shutdown order: stop accepting,
  let running requests finish within `http.shutdown_grace`, stop Wyoming,
  then stop the worker last, because draining requests still need it.
- If the worker dies, the gateway exits, so the container is restarted
  instead of answering "unavailable" forever.
- The worker is started last, after the listeners are bound, so a port
  conflict or a configuration error does not leave a worker loading a model.

The worker listens on a loopback port that an upstream-style `--port` value
is unlikely to collide with, and loads its model at startup, as upstream
does. Unlike upstream, the gateway is already listening during that time and
answers 503.

A single-engine image treats an unset `ASR_ENGINE` as "the engine of this
image", while upstream defaults to `openai_whisper`. Asking for an engine
the image does not contain fails at startup with a message that names the
image to use. The all-engines image behaves as upstream: `ASR_ENGINE` picks
the engine and defaults to `openai_whisper`. It runs one engine per
container start.

The all-in-one images run as root because upstream does: existing model
cache volumes are owned by root.

## Design principles

The code is organized around deep modules. The vocabulary below is used in
this precise sense throughout the project.

| Term | Meaning |
|---|---|
| module | anything with an interface and an implementation: a function, a type, a package, or the worker as a whole |
| interface | everything a caller must know to use a module correctly. Not only the signatures: also invariants, ordering constraints, error modes and required configuration |
| implementation | the code inside a module |
| seam | the place where a module's interface lives: a point where behavior can be changed without editing the callers |
| adapter | a concrete thing that satisfies an interface at a seam, for example the hub and a test fake are two adapters of `core.Service` |
| depth | how much behavior a caller gets per unit of interface it has to learn. A deep module hides a lot behind a small interface; a shallow one has an interface nearly as complex as its implementation |
| leverage | what callers gain from depth: one implementation serves many call sites and many tests |
| locality | what maintainers gain from depth: a change, a bug and the knowledge about them stay in one place instead of spreading over the callers |

The rules that follow from it:

- **Build deep modules.** Before exporting an identifier, ask whether it can
  stay inside. Fewer exported functions, simpler parameters and more
  complexity hidden behind them is the direction to push in. Depth is a
  property of the interface: a deep module may be composed of small
  swappable parts inside, as long as they are not part of its interface.
- **The deletion test.** Imagine the module deleted. If its complexity
  simply disappears, or moves unchanged into its only caller, the module was
  a pass-through and should be merged into that caller. If the complexity
  would reappear in several callers, the module earns its place.
- **The interface is the test surface.** Tests use the same interface as
  callers and assert on observable outcomes, so they survive internal
  changes. Wanting to test past the interface is a sign that the module has
  the wrong shape. Nothing is exported only because a test wants it.
- **One adapter is a hypothetical seam, two is a real one.** An interface
  type is introduced only where something really varies. A production
  implementation plus a test stand-in counts as two. There are no
  single-implementation interfaces kept "for flexibility".
- **Accept dependencies, do not create them.** A constructor receives the
  things it depends on (`New(cfg, service, decoder)`) instead of building
  them, which is what lets a test pass a fake. Functions return results
  rather than modify their arguments.

How this shows in the code:

- `core.Service` is deep. Three methods hide worker discovery and health,
  alias resolution, access control, capability checks, queueing with
  priorities, cancellation and usage accounting. That is why a frontend
  contains only protocol code.
- The seams are real ones. `core.Service` has the hub and the fakes in every
  frontend's tests. `core.AudioDecoder` has the ffmpeg decoder and test
  stand-ins. The worker API has the Python worker and the Go fake in
  `internal/hub/workertest`. The Python `Engine` interface has four engines.
- The worker API client is an internal seam of the hub
  (`internal/hub/internal/workerclient`): the Go toolchain itself prevents
  any other package from importing it.
- How a dependency is tested depends on its kind. Pure logic is tested
  directly. ffmpeg and the file system are local and cheap, so tests use the
  real thing. The worker is a separate process that the project owns, so it
  sits behind an interface with an in-memory adapter for tests.

## Known trade-offs

- **Audio is held in memory.** A job's PCM is one byte slice in the gateway
  and one array in the worker, about 115 MB per hour of audio on each side.
  Uploads are streamed to a spool file and never buffered, but the decoded
  audio is not streamed to the engine.
- **The queue is in memory.** A restart loses waiting jobs, and their
  clients see a failed request.
- **No job deadline.** Only the wait for the first line of a worker's
  response is bounded (`hub.first_byte_timeout`). A stuck engine holds its
  worker slot until the client gives up.
- **No preemption.** An interactive job waits for the running job to finish.
  With one worker and hour-long files that wait can be long; a separate
  worker for interactive use is the remedy.
- **State is per gateway process.** Usage totals, the rate limiter and the
  queue are not shared, so running several gateways side by side splits
  them.
- **The worker API is unauthenticated.** It relies on the network: loopback
  in the all-in-one image, a private network in the split setup.
- **Wyoming identity is the network address.** The protocol has no
  credentials, so access control is the source allowlist and one configured
  user for all connections.
- **Rate limiting uses fixed windows.** They are coarse, but need one map
  entry per address and no timers.
- **The request log is written synchronously.** A stalled disk stalls the
  completion of jobs.
- **Streaming depends on the engine.** whisperx returns everything at once
  and openai_whisper in 30-second steps, so an OpenAI streaming client gets
  its deltas in bursts with those engines.
- **Fidelity pins versions.** Engine libraries are pinned to the versions of
  the reference image, and the legacy pages and schemas are captures of it.
  Upgrading a library or following a new upstream release means capturing
  the reference again.
- **HTTP edge cases follow Go's `net/http`.** Where uvicorn and Go disagree
  below the application level (malformed requests, header order, request
  limits), whisper-hub behaves like Go.
- **The all-engines image is large and runs one engine per start.** Serving
  two engines at once takes two workers.
