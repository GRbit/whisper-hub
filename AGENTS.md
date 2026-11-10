# AGENTS.md

Orientation for coding agents and contributors working on this repository.
Read it before making changes: it covers where things live, how the pieces
talk to each other, the conventions of the codebase, what must not change,
and the commands that must pass before a change is done.

User documentation is [README.md](README.md). The reasons behind the
structure are in [docs/DESIGN.md](docs/DESIGN.md). The gateway-to-worker
protocol is specified in [docs/worker-api.md](docs/worker-api.md).

## What this is

A self-hosted speech-to-text service. A Go gateway (`whisper-hub`, one static
binary) serves three client protocols: the HTTP API of whisper-asr-webservice
("legacy"), OpenAI-compatible audio endpoints, and Wyoming for Home
Assistant. It authenticates callers, enforces per-user model access, decodes
audio with ffmpeg, schedules jobs by priority (pausing or restarting a
running lower one for a higher one) and accounts usage. Python workers
(`worker/`) run the Whisper engines (faster_whisper, whisperx,
openai_whisper, and a model-free `fake` for tests) behind an internal HTTP
API.

Stack: Go 1.25 (standard library `net/http` and `log/slog`; third-party
modules are limited to `golang.org/x/crypto`, `golang.org/x/term` and
`gopkg.in/yaml.v3`), Python 3.12 for the worker (uvicorn, python-multipart,
plus the engine libraries), Docker for images and end-to-end checks.

## Build, test, verify

The host needs Go, ffmpeg and Docker. Python packages are never installed on
the host: the worker, its tests and the client SDKs all run in containers.

```
make build        # bin/whisper-hub
make check        # gofmt check, go vet, go test -race ./...   (run before every commit)
make test-short   # same tests without the slow bcrypt and rate limiter ones
make image        # all-in-one image whisper-hub:$ENGINE (default faster_whisper)
make image-gpu    # CUDA all-in-one image whisper-hub:$ENGINE-gpu
make image-gateway image-worker image-worker-fake image-e2e
```

`ENGINE` is `faster_whisper`, `openai_whisper`, `whisperx` or `all`
(all-in-one only). Go tests that decode audio or start the composed gateway
skip themselves when ffmpeg is not in `PATH`, so a green run without ffmpeg
has tested less.

Worker tests run inside a test image:

```
docker build --build-arg ENGINE=fake --target test -t whisper-hub-worker:fake-test worker
docker run --rm whisper-hub-worker:fake-test
```

Build it with `ENGINE=faster_whisper` as well when touching
`legacy_writers.py`: that image additionally compares the copied writer code
with the installed library.

### End-to-end checks

They run real containers. The scripts are in `deploy/e2e/`.

| Target | What it checks | What it needs besides Docker |
|---|---|---|
| `make e2e-clients` | the openai Python SDK, the `wyoming` package, a replay of Home Assistant's speech-to-text client and zeroconf discovery, against the gateway image with a canned Go fake worker | Go and ffmpeg on the host |
| `make e2e-fake` | gateway image plus the Python fake worker: all three protocols, streaming, cancellation on disconnect, priority order and preemption (a Wyoming utterance pausing a long legacy job, and the timeout path with a fake that ignores the pause; read from the responses, the request log and the gateway log), authentication on and off | ffmpeg and python3 on the host |
| `make e2e-real` | the all-in-one image with model `tiny` on real recordings, including a Wyoming utterance overtaking a long legacy job whose body must stay that of an uninterrupted run | `VOICE_FILES`; ffmpeg and python3 on the host; network access to download the model |
| `make e2e-fidelity` | byte-for-byte comparison of legacy responses between the reference (built by `make image-reference`) and the all-in-one image, with whisper-hub's extensions applied explicitly | `VOICE_FILES`; network access to build the reference once; ffmpeg with AAC, MP3 and Opus encoders; python3 |
| `make e2e-packages` | every Python package of the three single-engine images is installed in `whisper-hub:all` at the same version | the four images already built (`make image ENGINE=...`); python3 |
| `make e2e` | `e2e-clients`, `e2e-fake`, `e2e-real`, `e2e-fidelity` | all of the above |
| `make e2e-engines` | builds all four all-in-one images, then `e2e-real` and `e2e-fidelity` for each single-engine image, `e2e-real` for `whisper-hub:all` with every `WHISPER_HUB_ENGINE` value, and `e2e-packages` | all of the above, and a lot of time and disk |

`ENGINE=<engine>` selects the image for `e2e-real` and `e2e-fidelity`
(`whisper-hub:$ENGINE`); `WHISPER_HUB_ENGINE`, when set, is passed to the container.
`REF_TWICE=1` makes the fidelity script first compare the reference with
itself and label the cases where it is not deterministic; use it for
openai_whisper.

Inputs the scripts take from the environment:

| Variable | Meaning |
|---|---|
| `VOICE_FILES` | a directory with two WAV recordings: `english-front-center.wav` (a copy is in `internal/legacy/testdata/audio/`; the checks expect the words "front center") and `german-20s.wav` (about 20 seconds of German speech; the checks expect language `de`). The fidelity script sends every `*.wav` in the directory |
| `REF_IMAGE` | the reference image; default from `deploy/e2e/reference/upstream.env`, as built by `make image-reference` |
| `CACHE_VOLUME` | Docker volume for downloaded models (default `whub-e2e-cache`) |
| `GATEWAY_IMAGE`, `WORKER_IMAGE`, `AIO_IMAGE`, `E2E_IMAGE` | override the images under test |

The reference is public upstream whisper-asr-webservice, `main` at the
commit pinned in `deploy/e2e/reference/upstream.env` (with its URL and the
image name). Anyone reproduces it with `make image-reference`, which runs
`docker build` on the git URL at that commit with upstream's own Dockerfile;
the container then runs as built. The only addition, for whisperx alone, is
`deploy/e2e/reference/sitecustomize.py`, the same torch allowlist the worker
uses, without which upstream's whisperx does not start. To move to a newer
upstream: change the commit, rebuild, run `capture.sh` and `make
e2e-fidelity`, and update `worker/requirements/constraints.txt` from the new
image's `pip freeze`.

`VOICE_FILES` has no default: `e2e-real` and `e2e-fidelity` stop with an
error without it, while `e2e-fake` then generates test tones and still runs.

CI (`.github/workflows/ci.yml`) runs the Go checks, the worker tests for
`fake` and `faster_whisper`, `e2e-clients` and `e2e-fake`. Nothing in CI
loads a real model. `.github/workflows/images.yml` builds every image and
publishes them on pushes to `master` and on `v*` tags; the tag rules are in
`.github/scripts/image-tags.sh`.

## Repo layout

Gateway (Go):

- `cmd/whisper-hub/` - the binary. `main.go`: subcommands (`serve`, `hash`,
  `apikey`, `config`, `healthcheck`, `version`) and signal handling.
  `gateway.go`: builds every module, binds the listeners, runs and shuts
  down in order. `gateway_test.go`: tests of the composed gateway against a
  fake worker.
- `internal/core/` - the contracts shared by all packages: `Service`,
  `AudioDecoder`, `Authorizer`, `UsageRecorder`, `Request`, `Result`,
  `Identity`, `Error` and its kinds, the Whisper language table.
- `internal/legacy/` - `/asr`, `/detect-language`, `/docs`, `/redoc`,
  `/openapi.json`, `/assets/`. `form.go` is a port of python-multipart,
  `pycompat.go` reproduces Python's JSON, float and URL behavior,
  `params.go` FastAPI's validation. `static/` and
  `testdata/reference/cases/` are captured from the reference by
  `testdata/reference/capture.sh`.
- `internal/openaiapi/` - `/v1/audio/transcriptions`,
  `/v1/audio/translations`, `/v1/models`; response formats, server-sent
  event streaming, OpenAI error objects.
  `internal/openaiapi/cmd/fakeserver` (the frontend on a canned service)
  and its `sdk_check.py` check it against the real openai SDK.
- `internal/wyoming/` - event codec, TCP server, per-connection state
  machine (`conn.go`), language mapping, zeroconf announcement (Linux only).
  `internal/wyoming/cmd/fakeserver` and `internal/wyoming/interop/` check
  it against the real `wyoming` package.
- `internal/auth/` - users file with reload (`users.go`), credential checks,
  login cache and rate limiter (`authn.go`), HTTP middleware
  (`middleware.go`), Wyoming identity (`wyoming.go`), usage totals and
  request log (`usage.go`), the admin endpoint (`admin.go`), password and
  API key hashing (`keys.go`).
- `internal/hub/` - the `core.Service` implementation: worker probing
  (`hub.go`), model resolution and access (`models.go`), the scheduler
  (`sched.go`), jobs and usage events (`service.go`).
  `internal/hub/internal/workerclient/` is the worker API client,
  importable only by the hub. `internal/hub/workertest/` is the fake worker
  used by Go tests and by `deploy/e2e/fakeworker`.
- `internal/audio/` - ffmpeg decoding through a spool file (`audio.go`) and
  pure Go PCM conversion and resampling (`raw.go`).
- `internal/config/` - the whole configuration: defaults, YAML file,
  upstream variables (`applyCompat`), `WHISPER_HUB_*`
  variables derived from the YAML paths (`env.go`), flags.
- `internal/workerproc/` - the worker as a child process, and the
  `WHISPER_HUB_ENGINE` check of the all-in-one images.

Worker (Python, `worker/`):

- `whisper_worker/app.py` - the worker API as a plain ASGI app.
- `whisper_worker/config.py` - environment variables.
- `whisper_worker/models.py` - model loading, eviction, one job per model.
- `whisper_worker/engines/` - `base.py` (the `Engine` interface, option
  parsing, the neutral segment shape) and one file per engine.
- `whisper_worker/legacy_writers.py` - upstream's writers for
  faster_whisper, copied. `model_resolver.py` - Hugging Face URLs.
- `requirements/` - one file per engine, `all.txt` for the all-engines
  image, `constraints.txt` with the versions the reference image installs
  (triton excepted: the version CUDA torch requires is used everywhere).
- `tests/` - pytest suite, run in the `test` stage of `worker/Dockerfile`.
- `scripts/fidelity.py` - compares a worker's rendered output with a running
  upstream service on the same raw PCM. `scripts/pause_check.py` - checks on
  a running worker that a paused and resumed job, and a job run during the
  pause, give exactly the output of uninterrupted runs.
  `scripts/cancel_check.py` - measures how soon a worker frees its model
  after a cancel at several points of a long job, and checks that the model
  still gives the same output afterwards.

Deployment:

- `deploy/Dockerfile`, `Dockerfile.gpu` - all-in-one images.
  `Dockerfile.gateway` - gateway only. `worker/Dockerfile`,
  `worker/Dockerfile.gpu` - worker only.
- `deploy/docker-compose.yml` (all-in-one), `docker-compose.auth.yml`
  (authentication overlay), `docker-compose.split.yml` (gateway and workers
  in separate containers).
- `deploy/whisper-hub.example.yaml` - every setting with its default.
  `deploy/auth/users.example.yaml` - users file template.
- `deploy/e2e/` - the end-to-end scripts.
- `.github/workflows/`, `.github/scripts/image-tags.sh` - CI and image
  publishing.

## How the pieces talk

```
HTTP client -> auth.Middleware -> legacy or openaiapi handler
                                    -> AudioDecoder (ffmpeg) -> PCM
                                    -> core.Service.Transcribe
Wyoming client -> wyoming server -> auth.WyomingIdentity
                                    -> AudioDecoder.FromRaw -> PCM
                                    -> core.Service.Transcribe

core.Service (internal/hub):
  resolve model name -> Authorizer.AllowModel -> drop or refuse unsupported options
  -> wait for a worker slot by priority (identity, else protocol default),
     pausing or canceling a lower running job -> workerclient: POST /v1/transcribe
  -> segments back through the onSegment callback -> Result
  -> UsageRecorder.Record (totals, request log, the Info log line)
```

Frontends depend only on `internal/core`. They do not authenticate, resolve
models, queue or account: the middleware and the hub do. Only
`internal/config` and `cmd/whisper-hub` import the other packages.

## Conventions

- ASCII only in code and comments: `->` instead of an arrow glyph. Never use
  the em-dash character anywhere, documentation and commit messages
  included; write a colon, a comma or " - ".
- Comments explain why, not what, in one or two lines. Comment what is
  deliberately not done (an omitted timeout, a skipped check) and anything
  surprising. Delete a stale comment rather than half-update it.
- Every exported Go identifier has a doc comment that starts with its name.
  Python docstrings follow the same habit.
- Go logging uses `log/slog`. Info is for lifecycle events and exactly one
  line per job, which the usage recorder writes (`job finished`); frontends
  and the hub log their own view of a job at Debug. Debug covers every step
  of an interaction: request parameters, outgoing HTTP calls, protocol
  events, queueing decisions. Warn is for failures an operator must see.
  Never log credentials or audio.
- Python logging uses the standard `logging` module with the same level
  discipline.
- Errors that cross a core interface are `*core.Error`: a kind plus a
  message that is safe to show to a client. Diagnostics (paths, subprocess
  output, library messages) go into the wrapped cause and the log, never
  into the message.
- Environment variables of whisper-hub start with `WHISPER_HUB_`: the
  gateway's are derived from the YAML paths, the worker's are
  `WHISPER_HUB_ENGINE` and `WHISPER_HUB_ENGINE_<NAME>` (no YAML section may
  be called `engine`), and `WHISPER_HUB_LOG_LEVEL` is read by both. The
  names of upstream whisper-asr-webservice are accepted only as aliases
  (`applyCompat`, `workerproc.EngineAlias`, `ALIASES` in
  `worker/whisper_worker/config.py`), listed in the README's drop-in table;
  our own files use the `WHISPER_HUB_` names, and the `WHISPER_HUB_` name
  wins when both are set.
- Packages do not read environment variables or files for configuration.
  They export a `Config` struct with `yaml` tags; the zero value of a field
  means its documented default.
- Tests go through a package's exported interface and assert on observable
  outcomes. Frontends are tested against a fake `core.Service` written in
  their own test files; the hub against `workertest`. Do not export
  something only because a test wants it.
- An interface type is introduced only where there are two real
  implementations, a test stand-in included.
- When the behavior of upstream, an OpenAI SDK or a Wyoming client is in
  question, check it against the real thing in a container instead of
  assuming, and record the result as a test or a captured case.
- Commit messages: a lowercase type prefix (`feat:`, `bugfix:`, `docs:`,
  `test:`, `ci:`, `perf:`), a short imperative subject, a blank line, and
  one short sentence saying why. The whole message is at most 196
  characters.

## What is frozen

Observable behavior does not change without the maintainer's explicit
approval. That covers:

- **The legacy API**: paths, parameter names and defaults, status codes,
  error bodies, response headers, content types and output bytes.
  Compatibility with upstream whisper-asr-webservice is guaranteed for
  whisper-hub major versions 0 and 1; from 2.0 on it may be dropped, and
  the reference checks then no longer need to run. The promise is API
  compatibility plus whisper-hub's extensions (listed in the README,
  "Extensions to upstream", and applied in
  `internal/legacy/reference_test.go` and `deploy/e2e/fidelity.py`), not
  byte-for-byte identity: the byte-level comparison is a verification
  tool, and a change that makes it differ still needs the maintainer's
  approval. Guards: `go test ./internal/legacy` replays the recorded reference
  exchanges in `testdata/reference/cases/` and compares status, headers and
  body; `make e2e-fidelity` compares whole responses against a running
  reference. A legitimate change means running
  `internal/legacy/testdata/reference/capture.sh` against the reference
  again; never edit `internal/legacy/static/` or the recorded cases by hand.
- **Deployment compatibility**: the upstream environment variables and
  flags that `internal/config` accepts, and the model cache
  path of the images.
- **whisper-hub's own surface**: configuration keys (the `WHISPER_HUB_*`
  names are derived from the `yaml` tags, so renaming a tag renames a
  variable), subcommands and flags, the users file format, the field names
  of the usage totals file and the request log, `/healthz`, `/_hub/usage`,
  image names and tags, and error strings clients may match on.
- **Engine library versions** in `worker/requirements/`: they are the
  reference image's, because output fidelity depends on them.

The worker API is internal and may change, provided the Python worker, the
Go client, the Go fake and `docs/worker-api.md` change together.

## Making common changes

- **New config setting**: add the field, with a `yaml` tag and a doc comment
  that states the default, to the owning package's `Config` struct (or to
  `HTTP` or `Log` in `internal/config/config.go` for a setting of the binary
  itself). Resolve the default inside the package, in the `Resolved` method
  of its `Config` that the constructor goes through; `whisper-hub config`
  prints through the same method (`TestYAMLShowsResolvedDefaults` fails on
  a setting that still prints as zero, unless its default really is zero).
  The environment variable appears by itself. Add the key with
  its default to `deploy/whisper-hub.example.yaml`
  (`TestExampleConfigParses` fails on a key the code does not know), and to
  the README if users need it. If it replaces an upstream variable, map
  that variable in `applyCompat` and extend
  `internal/config/config_test.go`.
- **New engine backend**: add `worker/whisper_worker/engines/<name>.py` with
  a subclass of `Engine`: set `name` and `capabilities`, implement `load`,
  `transcribe` and `detect_language`. In `transcribe`, call `ctx.info(...)`
  first, `ctx.segment(...)` per segment, `ctx.check_cancelled()` whenever
  the library returns control, and fill `Done.rendered` when
  `opts.legacy_output` is set. `ctx.check_cancelled()` is also the pause
  point: set `pausable = True` only after checking with
  `worker/scripts/pause_check.py` that another job on the same loaded model
  between two such calls leaves both outputs unchanged, and call it only
  where no shared state of the model is half-changed (see the whisperx
  options swap). Where the library runs long without returning control,
  look for a per-batch or per-segment call that a cancel-only hook can
  wrap (see `engines/whisperx.py`): guard it so an unexpected library shape
  keeps working without the hook and logs a warning. Measure with
  `worker/scripts/cancel_check.py`. Add the engine to the pause and cancel
  tables in `docs/worker-api.md`. Register the engine in `engines/__init__.py`
  (`create` and `cuda_probe`) and in `ENGINES` in `config.py`. Add
  `worker/requirements/<name>.txt` (starting with `-r base.txt`) and the
  name to the `case` statements of the four engine Dockerfiles. On the
  gateway side add it to `Engines` in `internal/workerproc/workerproc.go`
  (the real engines come first, `fake` last), to `openAPIFile` in
  `internal/legacy/static.go` if upstream has a schema for it, and to
  `engineAttribution` in `internal/wyoming/conn.go`. Then the build matrix
  in `.github/workflows/images.yml`, the loops of `e2e-engines` in the
  Makefile, and the capability tables in `docs/worker-api.md`.
- **New protocol frontend**: create `internal/<name>/` importing only
  `internal/core`. Export a `Config` and a constructor that takes it plus a
  `core.Service` and, if it receives audio, a `core.AudioDecoder`. An HTTP
  frontend exposes `Register(mux *http.ServeMux)` and reads the caller with
  `core.IdentityFrom(r.Context())`; a non-HTTP one takes an identify
  function like `wyoming.IdentifyFunc` and stores the identity with
  `core.WithIdentity`. Pass `Request.Model` as received, set `Protocol` (add
  a constant in `internal/core/core.go`, and a default to `hub.Priorities`
  with its case in `priorityOf` in `internal/hub/sched.go`) and `Endpoint`;
  frontends never set a priority. Map
  each `core.Error` kind to the protocol's own error. Add the `Config` to
  `config.Config`, construct and register the frontend in `newGateway` in
  `cmd/whisper-hub/gateway.go`, add its paths to the default
  `protected_paths` in `internal/auth/config.go`, and pass a rejection
  writer to `auth.New` if its clients expect a specific 401 body. Test it
  against a fake `core.Service`, and add a composed test in
  `cmd/whisper-hub/gateway_test.go`.
- **Worker API change**: change `worker/whisper_worker/app.py` (and
  `engines/base.py` for options or events),
  `internal/hub/internal/workerclient/`, `internal/hub/workertest/` and
  `docs/worker-api.md` in the same commit. The Go fake is strict on purpose,
  so a client that drifts from the API fails loudly.
- **New image variant**: add it to the matrix in
  `.github/workflows/images.yml`; tags follow from
  `.github/scripts/image-tags.sh`. Update the image tables in the README.
