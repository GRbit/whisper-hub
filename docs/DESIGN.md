# whisper-hub design

One speech-to-text service that speaks three client protocols at once:

1. **Legacy**: the HTTP API of `ahmetoner/whisper-asr-webservice` (`/asr`,
   `/detect-language`, `/docs`), 100% compatible. Existing clients and
   deployments (same port, same env vars) must keep working unchanged.
2. **OpenAI**: `/v1/audio/transcriptions`, `/v1/audio/translations`,
   `/v1/models`. Compatible at the level of `whisper-1`; full parity with
   the gpt-* transcription models is a non-goal, and so is the Realtime API.
3. **Wyoming**: TCP protocol used by Home Assistant (default port 10300).

Plus authentication, per-user model access and usage logging built in.

## Architecture

```
clients --> gateway (Go, one static binary)            --> engine worker(s) (Python)
            :9000  HTTP  legacy + OpenAI + /healthz         :9100 internal HTTP
            :10300 TCP   Wyoming                            faster-whisper | whisperx |
            auth, model ACL, usage log                      openai-whisper | fake
            ffmpeg decode -> PCM s16le 16 kHz mono
            priority queue per worker
```

- The gateway owns everything protocol-shaped. Workers only run models.
- Audio is decoded once, in the gateway. Workers receive raw PCM and need
  no ffmpeg. This also gives the gateway the audio duration for accounting.
- A worker serves one engine and one or more models. Several workers can be
  attached (for example faster-whisper and whisperx in separate containers).
- Shipping: an all-in-one image per engine (`deploy/Dockerfile`, gateway +
  worker, drop-in replacement for the upstream image), `whisper-hub:all`
  with every engine installed (`ENGINE=all`, `ASR_ENGINE` picks one per
  container start, unset means `openai_whisper` as upstream), a gateway-only image
  (`deploy/Dockerfile.gateway`), worker images (`worker/Dockerfile`), and
  compose files for both setups. Both Go images ship upstream's ffmpeg build
  (`onerahmet/ffmpeg:n7.1`), so uploads decode to the same PCM as upstream.
- Supervision in the all-in-one image: the gateway is PID 1 and starts the
  worker as its child (`internal/workerproc`, configured by `worker.command`).
  Signals reach the process that knows the shutdown order (drain HTTP and
  Wyoming, then stop the worker); the gateway exits when the worker dies,
  so the container restarts. On a single-engine image `ASR_ENGINE` unset
  means the image's engine and a different one fails at startup naming the
  image that has it.
- CI (`.github/workflows`): `ci.yml` (Go checks, worker pytest, fake-engine
  end-to-end runs) and `images.yml` (every image including the CUDA ones,
  pushed to ghcr.io/grbit on master and v* tags, layer cache on ghcr.io).

## Repository layout and ownership

| Path | Owner | Contents |
|------|-------|----------|
| `internal/core/` | lead (frozen) | Shared contracts. Do not edit; report what you need changed |
| `internal/legacy/` | legacy | `/asr`, `/detect-language`, `/`, `/docs`, `/openapi.json` |
| `internal/openaiapi/` | openai | `/v1/audio/*`, `/v1/models` |
| `internal/wyoming/` | wyoming | Wyoming codec and TCP server |
| `internal/auth/` | auth | users, HTTP auth middleware, model ACL, usage accounting, request log |
| `internal/hub/` | runtime | `core.Service` impl: worker registry, model resolution, scheduler, usage; the Worker API client is internal to it (`internal/hub/internal/workerclient`), the fake worker for tests is `internal/hub/workertest` |
| `internal/audio/` | runtime | `core.AudioDecoder`: ffmpeg for files, pure Go PCM conversion |
| `internal/config/` | integration | the whole configuration: YAML file, `WHISPER_HUB_*` env, upstream and sidecar variables, flags |
| `internal/workerproc/` | integration | the worker child process of the all-in-one image |
| `worker/` | worker | Python engine worker, its Dockerfiles and tests |
| `cmd/whisper-hub/` | integration | the binary: composes the modules (`serve`), `hash`, `apikey`, `config`, `healthcheck` |
| `deploy/` | integration | Dockerfiles, compose files, example config, users file template, end-to-end scripts (`deploy/e2e`) |

Each owner works in its own git worktree and edits only its own paths.
`go.mod`/`go.sum` may be changed locally to add dependencies; list the
additions in the final report so they can be merged.

## Go package rules

- Depend on `internal/core` only (plus stdlib and a minimal set of third
  party modules). Frontends never import each other, the runtime or auth.
  Only the composition layer (`cmd/whisper-hub`, `internal/config`) imports
  the modules themselves; `main` composes and holds no logic.
- Export a `Config` struct with `yaml` tags and documented defaults, and a
  constructor taking that config plus `core` interfaces. Do not read
  environment variables or files for configuration inside the package; the
  integration phase composes the root config.
- HTTP frontends expose `func (*T) Register(mux *http.ServeMux)` (Go 1.22+
  method patterns) or an `http.Handler`. They read the caller with
  `core.IdentityFrom(ctx)`; they do not authenticate.
- Frontends pass `Request.Model` as received. Alias resolution, ACL, queueing
  and usage records all happen inside `core.Service`.
- Map `core.Error` kinds to protocol errors; never leak `Error.Err` text.
- Test against a fake `core.Service` written in your own `_test.go` files.

## Design principles: deep modules

Vocabulary and rules follow the `codebase-design` skill. Use the terms
module, interface, seam, adapter, depth, leverage, locality.

- Build deep modules: a lot of behavior behind a small interface. Before
  adding an exported identifier ask whether it can stay inside. The
  interface is everything a caller must know: signatures, but also
  invariants, ordering, error modes and required config. Keep all of it small.
- The deletion test: if deleting a module only moves its code into the one
  caller, it was a pass-through; merge it. Do not split a package into
  helpers that have a single caller and no behavior of their own.
- The interface is the test surface. Test through the exported interface
  and assert on observable outcomes, so tests survive internal refactors.
  Wanting to test past the interface means the module has the wrong shape.
  Do not export something only because a test wants it.
- One adapter is a hypothetical seam, two is a real one. Introduce a Go
  interface only where something really varies (production plus a test
  stand-in counts). No single-implementation interfaces "for flexibility".
- Internal seams stay internal: a module may be composed of swappable parts,
  but they are not part of its interface.
- Accept dependencies, do not create them; return results rather than
  mutate arguments.
- Dependencies by category: in-process logic is tested directly; ffmpeg and
  the filesystem are local-substitutable, test with the real thing; the
  engine worker is remote but owned, so it sits behind a port
  (`core.Service` for frontends, the worker HTTP API for the hub) with an
  in-memory or httptest adapter for tests.

## Model selection and access

- Workers report their models. The hub exposes them by id, plus aliases from
  config. `whisper-1` is an alias of the default model out of the box.
- Unknown model names fall back to the default model by default (many OpenAI
  clients hardcode names); a config switch makes them an error instead.
- Per-user `models` patterns in the users file decide access. A user with no
  patterns may use only the default model. A request for a known model the
  user is not allowed to use fails with `KindForbidden`; it never silently
  falls back.
- `/v1/models` and Wyoming `info` list only what the caller may use.
- Wyoming has no credentials: connections map to a configured user, with an
  optional source CIDR allowlist.

## Authentication

- Off when no users file is configured (everything anonymous, as upstream).
- Accepted credentials: HTTP Basic (bcrypt password), `Authorization: Bearer
  <api key>`, and `X-Api-Key: <api key>`.
- Protected: `/asr`, `/detect-language`, `/v1/*`. Public: `/`, `/docs`,
  `/openapi.json`, `/assets/*`, `/healthz`.
- Every job produces a usage record (user, protocol, endpoint, model, audio
  seconds, queue and processing time, status) and a log line.

## Worker API (internal, no auth)

HTTP/1.1 on port 9100. All JSON is UTF-8.

### `GET /healthz`
`200 {"status":"ok"}` once the process can accept jobs.

### `GET /v1/info`
```json
{
  "engine": "faster_whisper",
  "version": "worker and library versions, free form",
  "device": "cpu",
  "models": [{"id": "tiny", "default": true, "loaded": true}],
  "languages": ["af", "am", "..."],
  "capabilities": {"word_timestamps": true, "vad_filter": true, "diarize": false, "translate": true}
}
```
Exactly one model has `"default": true`.

### `POST /v1/transcribe`
`multipart/form-data` with two parts, in this order:

- `options`: `application/json`
  ```json
  {
    "model": "tiny",            // optional, default model when absent
    "task": "transcribe",       // or "translate"
    "language": null,           // Whisper code or null for auto
    "prompt": null,
    "temperature": null,        // null = engine default
    "vad_filter": false,
    "word_timestamps": false,
    "diarize": false, "min_speakers": null, "max_speakers": null,
    "legacy_output": null       // "txt" | "vtt" | "srt" | "tsv" | "json" | null
  }
  ```
- `audio`: `application/octet-stream`, raw PCM s16le mono 16 kHz.

Response `200 application/x-ndjson`, one JSON object per line, flushed as
produced:

```
{"type":"info","model":"tiny","engine":"faster_whisper","language":"en","language_probability":0.98,"duration":12.34}
{"type":"segment","segment":{"id":0,"seek":0,"start":0.0,"end":2.5,"text":" Hello","tokens":[1,2],"temperature":0.0,"avg_logprob":-0.2,"compression_ratio":1.1,"no_speech_prob":0.01,"words":[{"word":" Hello","start":0.0,"end":0.4,"probability":0.9}],"speaker":null}}
{"type":"done","text":" Hello","language":"en","rendered":"Hello\n"}
```

- `info` comes first, `done` last. `words` and `speaker` may be null/absent.
- Engines that cannot stream (whisperx, openai-whisper) emit all segments
  right before `done`.
- `rendered` is present only when `legacy_output` was set. It is the exact
  body the upstream service would return from `/asr` for this engine and
  output format, produced with the same code upstream uses.
- A failure after the stream started is a final line
  `{"type":"error","code":"...","message":"..."}` instead of `done`.
- A failure before any output is an HTTP error with body
  `{"error":{"code":"...","message":"..."}}`: `400 bad_request`,
  `404 model_not_found`, `503 unavailable`, `500 internal`.
- When the client closes the connection the worker stops the job as soon as
  the engine allows.
- The worker runs one job at a time per model and queues the rest; the
  gateway normally never sends more than one job per worker at once.

### `POST /v1/detect-language`
Same multipart shape; `options` is `{"model": null}`. Response
`200 {"language":"en","confidence":0.98,"model":"tiny","engine":"faster_whisper"}`.
Only the first 30 seconds are used, as upstream does.

## Legacy compatibility reference

- Upstream source: `/path/to/whisper-asr-webservice`
  (upstream; branches `main` and `main` matter. Read other
  branches with `git show <branch>:<path>`; do not switch branches there
  and do not modify that repository).
- A built reference image exists locally: `whisper-asr-webservice:ec32cd5`. Use it
  to capture ground-truth responses rather than assuming behavior.
- Frozen observable behavior: paths, parameter names and defaults, status
  codes, error bodies, response headers, content types, output bytes, and
  the environment variables documented in `docs/environmental-variables.md`.

## Conventions (apply to all code in this repository)

- Commit only on your own worktree branch and only your own paths. The
  whole commit message is at most 196 characters: `type: short imperative
  subject`, a blank line, one short sentence on why. Never run `git push`.
  Never use `git -C`; `cd` instead.
- ASCII only in code and comments: `->` not an arrow glyph. Never use the
  em-dash character anywhere, including docs and reports; use " - ", a colon
  or a comma.
- Comments explain why, not what; one or two lines. Exported identifiers get
  a doc comment starting with the identifier name.
- Go logging: `log/slog`. Info for lifecycle and one line per job, written
  by the usage recorder (frontends and the hub log jobs at Debug);
  Debug for every step of an interaction (request parameters, outgoing HTTP
  calls, protocol events, queueing). Never log credentials or audio.
- Python: stdlib `logging` with the same level discipline.
- Do not install software on the host. Go, git, docker, ffmpeg and python3
  are available; everything else (pip packages, other toolchains, client
  SDKs) runs in docker. The machine is slow: use the smallest models
  (`tiny`) and avoid pulling large images when an existing one will do.
  A Hugging Face token is in `/path/to/.env`
  if a download needs it; never print or copy it into the repository.
- Go verification before reporting: `gofmt -l`, `go vet ./...`,
  `go test -race ./...` for your packages.
- Claims need evidence: when behavior of upstream, the OpenAI SDK, or a
  Wyoming client is in question, check it against the real thing in docker.
- Shell `cd` does not persist between your commands: use absolute paths or
  `cd <dir> && <cmd>`.

## Final report format (for module owners)

Keep it under 60 lines: public API of your package (signatures), config
fields, dependencies added, what was verified and how (commands, results),
known gaps, and anything you need changed in `internal/core`.
