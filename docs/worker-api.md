# Worker API

The worker API is the internal HTTP interface between the whisper-hub
gateway and an engine worker. The gateway is the only intended client. It
sends decoded audio and engine-neutral options; the worker runs the model and
streams the transcript back.

This is not a public API. It has no authentication and no compatibility
promise: the gateway and the worker are released together. Keep the worker
port on a private network (in the all-in-one image it listens on loopback
only) and send client traffic to the gateway.

Where it is implemented:

| Side | Location |
|---|---|
| Server (Python) | `worker/whisper_worker/app.py`, option parsing in `worker/whisper_worker/engines/base.py` |
| Client (Go) | `internal/hub/internal/workerclient/` |
| In-process fake for Go tests | `internal/hub/workertest/` |

A change to the API has to be made in all three.

## Conventions

- HTTP/1.1. The worker listens on `WHISPER_HUB_ENGINE_HOST`:`WHISPER_HUB_ENGINE_PORT`, by default
  `0.0.0.0:9100`. The all-in-one images use `127.0.0.1:19100`.
- All JSON is UTF-8.
- Audio is always raw PCM: signed 16-bit little-endian, mono, 16 kHz, no
  header. The worker never decodes containers or codecs and needs no ffmpeg.
- Numbers are finite. Where an engine produces NaN or infinity the worker
  sends `0`; where an engine has no value it sends `null`.

| Method and path | Purpose |
|---|---|
| `GET /healthz` | liveness |
| `GET /v1/info` | engine, models, languages, capabilities |
| `POST /v1/transcribe` | run a transcription or translation, streamed |
| `POST /v1/detect-language` | identify the spoken language |
| `POST /v1/pause` | pause a running transcription for as long as the request stays open |

Any other path answers `404` with code `not_found`; a known path with the
wrong method answers `405` with code `method_not_allowed`. Both use the
error body described under [Errors](#errors).

## `GET /healthz`

`200 {"status":"ok"}` as soon as the process accepts requests.

With `WHISPER_HUB_ENGINE_PRELOAD=true` the worker loads its default model before it
starts listening, so a successful `/healthz` then also means that the default
model is in memory. Without preload the worker is healthy immediately and
loads a model on the first job that needs it.

## `GET /v1/info`

```json
{
  "engine": "faster_whisper",
  "version": "whisper-worker 0.1.0; faster-whisper 1.2.1, ctranslate2 4.8.1",
  "device": "cpu",
  "models": [{"id": "tiny", "default": true, "loaded": false}],
  "languages": ["af", "am", "ar"],
  "capabilities": {"word_timestamps": true, "vad_filter": true, "diarize": false, "translate": true}
}
```

| Field | Meaning |
|---|---|
| `engine` | `faster_whisper`, `whisperx`, `openai_whisper` or `fake` |
| `version` | free-form text: the worker version, then the engine library versions |
| `device` | the configured device, `cpu` or `cuda` |
| `models` | the models this worker serves, in `WHISPER_HUB_ENGINE_MODELS` order. Exactly one has `"default": true` (the first). `loaded` tells whether the model is in memory right now |
| `languages` | the Whisper language codes; the full list is in `worker/whisper_worker/languages.py` |
| `capabilities` | the optional features of the engine, see below |

Capabilities by engine:

| Engine | `word_timestamps` | `vad_filter` | `diarize` | `translate` |
|---|---|---|---|---|
| `faster_whisper` | true | true | false | true |
| `openai_whisper` | true | false | false | true |
| `whisperx` | true | false | true only when `HF_TOKEN` is set | true |
| `fake` | true | true | false | true |

The gateway polls `/healthz` and `/v1/info` to learn which models exist and
whether the worker is usable. It uses `capabilities` to decide, before
sending a job, whether an option the engine lacks is dropped or the request
is refused. The worker itself never rejects such an option; it ignores it.

## `POST /v1/transcribe`

### Request

`multipart/form-data` with two parts:

1. `options`: `application/json`, the job options.
2. `audio`: `application/octet-stream`, the raw PCM.

The gateway sends them in this order, and the Go test fake requires this
order. The Python worker accepts either order, and treats a missing or empty
`options` part as `{}`.

```json
{
  "model": "tiny",
  "task": "transcribe",
  "language": null,
  "prompt": null,
  "temperature": null,
  "vad_filter": false,
  "word_timestamps": false,
  "diarize": false,
  "min_speakers": null,
  "max_speakers": null,
  "legacy_output": null,
  "job_id": "3f2a91c0-7-1"
}
```

| Option | Type | Default | Meaning |
|---|---|---|---|
| `model` | string or null | the worker's default model | one of the ids in `/v1/info` |
| `task` | string | `transcribe` | `transcribe`, or `translate` (to English) |
| `language` | string or null | null: detect | a Whisper language code |
| `prompt` | string or null | null | initial prompt for the decoder |
| `temperature` | number or null | null: the engine's default schedule | non-negative |
| `vad_filter` | boolean | false | voice activity filter (faster_whisper) |
| `word_timestamps` | boolean | false | fill `words` in each segment |
| `diarize` | boolean | false | label speakers (whisperx with `HF_TOKEN`) |
| `min_speakers`, `max_speakers` | integer or null | null | non-negative; diarization hints |
| `legacy_output` | string or null | null | `txt`, `vtt`, `srt`, `tsv` or `json`: also render the transcript in that whisper-asr-webservice format, see `rendered` below |
| `job_id` | string or null | null | the name `POST /v1/pause` uses for this job; chosen by the gateway, unique per attempt. The gateway omits it when it is empty |

Rules the worker applies:

- Unknown keys are ignored. An absent key, `null` and an empty string all
  mean the default.
- A value of the wrong type, an unknown `task`, `language` or
  `legacy_output`, or `options` that is not a JSON object, is
  `400 bad_request`.
- A `model` this worker does not serve is `404 model_not_found`.
- A missing `audio` part, an empty one, or one with an odd number of bytes is
  `400 bad_request`.

### Response

`200` with `Content-Type: application/x-ndjson`: one JSON object per line,
each line written and flushed as soon as it is known. The status line is sent
together with the first event, so everything that fails earlier is a plain
HTTP error (see [Errors](#errors)).

```
{"type":"info","model":"tiny","engine":"faster_whisper","language":"en","language_probability":0.98,"duration":12.34}
{"type":"segment","segment":{"id":1,"seek":0,"start":0.0,"end":2.5,"text":" Hello","tokens":[50364,2425],"temperature":0.0,"avg_logprob":-0.2,"compression_ratio":1.1,"no_speech_prob":0.01,"words":[{"word":" Hello","start":0.0,"end":0.4,"probability":0.9,"speaker":null}],"speaker":null}}
{"type":"done","text":" Hello","language":"en","rendered":"Hello\n"}
```

The stream is one `info` event, then zero or more `segment` events, then
exactly one `done` or `error` event. `paused` and `resumed` events may come
anywhere before the last line, also before `info`:

```
{"type":"paused"}
{"type":"resumed"}
```

`paused` means the job stopped at a pause point because a pause request
names it, and gave its model to the next job; `resumed` means it has the
model again and continues. Between the two the stream is silent. See
[`POST /v1/pause`](#post-v1pause).

`info`:

| Field | Meaning |
|---|---|
| `model`, `engine` | the model id and engine that run the job |
| `language` | the requested language, or the detected one |
| `language_probability` | detection confidence; `null` for openai_whisper and whisperx, which do not report it |
| `duration` | audio length in seconds |

`segment`, inside the `segment` object:

| Field | Meaning |
|---|---|
| `id` | the engine's own segment number; do not assume where it starts |
| `seek`, `tokens`, `temperature`, `avg_logprob`, `compression_ratio`, `no_speech_prob` | Whisper decoding details. whisperx does not have them: it sends `seek` 0, `tokens` `[]` and `null` for the rest |
| `start`, `end` | seconds from the start of the audio |
| `text` | segment text as the engine produced it, usually with a leading space |
| `words` | `null`, or a list of `{"word", "start", "end", "probability", "speaker"}` when word timestamps were produced |
| `speaker` | `null`, or the speaker label when the job was diarized |

`done`:

| Field | Meaning |
|---|---|
| `text` | the whole transcript |
| `language` | the language of the transcript |
| `rendered` | present only when `legacy_output` was set: the exact body the reference whisper-asr-webservice returns from `/asr` for this engine and output format, produced with the same writer code. The gateway passes it to legacy clients unchanged |

How promptly segments arrive depends on the engine:

| Engine | Segments are sent |
|---|---|
| `faster_whisper`, `fake` | one by one, as each is decoded |
| `openai_whisper` | after each 30-second window of audio; all at the end if a library update breaks the progress hook the worker relies on |
| `whisperx` | all together, right before `done` (alignment needs the whole transcript) |

A failure after the stream has started replaces `done` with a final line:

```
{"type":"error","code":"internal","message":"transcription failed"}
```

What the gateway does with the stream: it ignores blank lines and unknown
event types, treats a stream that ends without `done` or `error` as an
internal failure, and rejects a line longer than 64 MiB. Any line ends the
first-byte wait (`hub.first_byte_timeout`), so a job paused before its
`info` line is not taken for a stuck worker.

## `POST /v1/detect-language`

Same multipart shape. `options` is `{"model": "tiny"}`; `model` may be
`null` or absent for the default model, and other keys are ignored.

```json
{"language": "en", "confidence": 0.98, "model": "tiny", "engine": "faster_whisper"}
```

Only the first 30 seconds of audio matter, as in whisper-asr-webservice: the
gateway sends no more than that, and the engines pad or trim to 30 seconds.
The response is a single JSON document, not a stream, so every failure is an
HTTP error.

## `POST /v1/pause`

```json
{"job_id": "3f2a91c0-7-1", "model": "tiny"}
```

| Field | Meaning |
|---|---|
| `job_id` | the `job_id` of a transcription; it need not have arrived yet |
| `model` | the model the gateway will run while this job is paused, or null |

The pause lasts exactly as long as this request stays open:

1. The worker answers `200` with `Content-Type: application/x-ndjson` and
   one line `{"status":"pausing"}`, and keeps the response open. That only
   means the pause is registered.
2. At its next pause point the job stops, gives its model to the next job
   waiting for it, and its transcribe stream sends `paused`. A job that
   has not reached the worker yet pauses at its first pause point, before
   any output.
3. When the gateway closes the request, the job takes its model back, ahead
   of jobs that wait for it, its stream sends `resumed`, and it continues.
   Closing the request before the job paused withdraws the pause.
4. When the job ends (done, failed, or canceled while paused), the worker
   ends this response.

A paused job keeps its thread, its audio and its engine state; a model with
a paused job is never unloaded. Canceling a paused job (closing its
transcribe connection) ends it at once.

Errors, before anything else is sent:

| Status | `code` | When |
|---|---|---|
| 400 | `bad_request` | the body is not a JSON object with a non-empty `job_id` string, or `model` is not a string or null |
| 409 | `not_pausable` | the engine cannot pause at all, or `model` is not loaded and could only be loaded by unloading the paused job's model (`WHISPER_HUB_ENGINE_MAX_LOADED_MODELS`) |

The gateway treats `409` like a pause that did not take effect in time. Only
transcriptions pause; `POST /v1/detect-language` has no pause point.

### Pausing

A pause point is `ctx.check_cancelled()` in the engine adapter, the same
place where cancellation is noticed. Each engine checks once before it
starts, so a job can always pause before doing any work, and then:

| Engine | Pauses | Why that is safe |
|---|---|---|
| `faster_whisper` | between segments | the segment generator keeps its decoding state to itself; another call on the same `WhisperModel` runs in between |
| `openai_whisper` | between 30-second windows, and once after decoding | the decoder's key-value cache hooks are installed and removed within each window |
| `whisperx` | between its transcribe, align and diarize stages | the pipeline's tokenizer and options belong to the call in progress during transcribe and are reset when it returns; there is no pause point inside it, so on a long file a pause usually takes effect only after most of the work, and the gateway's timeout path (cancel and restart) applies instead |
| `fake` | before each segment, unless the prompt has the directive `pause=ignore` | |

Checked per engine with model `tiny` on the CPU images, using
`worker/scripts/pause_check.py`: a 2-minute recording runs uninterrupted, then again
paused once at its first pause point and once one second in, while a short
recording runs to the end in the gap. Both streams, `rendered` included
(`legacy_output` json and srt, and with `word_timestamps` for
faster_whisper and openai_whisper), were identical to their uninterrupted
runs for all three engines; whisperx paused 2.5 s after the request, at the
end of its transcribe stage. Output with temperature fallback depends on
the random state of the process in openai_whisper with or without a pause;
the check also runs the long recording twice without pausing and reports
when the engine is not deterministic on it.

## Errors

A failure before any output is an HTTP error with this body:

```json
{"error": {"code": "bad_request", "message": "options.task must be one of transcribe, translate"}}
```

| Status | `code` | When |
|---|---|---|
| 400 | `bad_request` | malformed multipart body, invalid options, invalid audio payload |
| 404 | `model_not_found` | the worker does not serve the requested model |
| 503 | `unavailable` | the job cannot run now, for example the model failed to load |
| 409 | `not_pausable` | `POST /v1/pause` only, see there |
| 500 | `internal` | the engine failed |
| 404 | `not_found` | unknown path |
| 405 | `method_not_allowed` | known path, wrong method |

The same `code` values appear in the `error` line of a transcribe stream.

`message` is safe to show to a client: the worker never puts file paths,
library messages or tracebacks into it. Those go to the worker log.

The gateway turns a `code` into its own error kind of the same name and
decides what the client sees. Only the `bad_request` message is forwarded to
clients (cut at 200 characters); for every other code the client gets a
fixed gateway message and the worker's text stays in the gateway log. An
unknown code counts as `internal`. If a response has no parsable error body,
the gateway goes by the status: 400 and 422 mean `bad_request`, 404
`model_not_found`, 503 and 429 `unavailable`, anything else `internal`.

## Concurrency and model loading

- Each model runs one job at a time. Further jobs for the same model wait in
  the worker, in arrival order; a paused job does not count, and a resumed
  one goes first. Jobs for different models run concurrently.
- The gateway normally never makes a job wait there: it sends at most
  `max_in_flight` running jobs per worker at once (default 1), not counting
  paused ones, and keeps the rest in its own priority queue.
- A model is loaded on the first job that needs it, unless `WHISPER_HUB_ENGINE_PRELOAD`
  loaded the default model at startup. At most `WHISPER_HUB_ENGINE_MAX_LOADED_MODELS`
  models are in memory; to load another one the worker unloads the idle model
  that was used least recently, or waits until one becomes idle.
- With `WHISPER_HUB_ENGINE_MODEL_IDLE_TIMEOUT` greater than 0, a model that has not been used
  for that many seconds is unloaded.
- A model that cannot be loaded fails the job with `503 unavailable`.

## Cancellation

The gateway cancels a job by closing the connection: when its own client
disconnects, when a streaming consumer fails, or when its shutdown grace
period ends. The worker notices the disconnect and stops the job as soon as
the engine gives control back:

| Engine | Stops |
|---|---|
| `faster_whisper` | between segments |
| `openai_whisper` | between 30-second windows |
| `whisperx` | between its transcribe, align and diarize stages |
| `fake` | at once |

A job that was still waiting for its model is dropped without running. A
paused job stops at once.

## What the gateway expects in time

These limits are gateway settings (`hub.*`), listed here because they define
how a worker must behave to be considered healthy:

| Setting | Default | Applies to |
|---|---|---|
| `hub.connect_timeout` | 5s | opening a TCP connection to the worker |
| `hub.probe_timeout` | 30s | one `/healthz` or `/v1/info` call |
| `hub.first_byte_timeout` | 10m | the wait for the first line of a transcribe response, which has to cover loading the model. `/detect-language` gets the longer of this and `hub.probe_timeout` as its overall deadline |
| `hub.poll_interval` | 15s | how often a healthy worker is probed again |
| `hub.retry_min`, `hub.retry_max` | 1s, 30s | bounds of the exponential backoff between probes of a worker that is down or still loading |
| `hub.fail_threshold` | 2 | consecutive failed probes before a healthy worker is taken out of rotation |
| `hub.preempt_timeout` | 5s | from opening `POST /v1/pause` until the job's stream reports `paused`; after it the gateway cancels the job to restart it later |

A probe is `/healthz` followed by `/v1/info`. There is deliberately no
overall deadline for a transcription: a long recording can take an hour.
