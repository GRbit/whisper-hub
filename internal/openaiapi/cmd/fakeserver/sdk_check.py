"""Checks the OpenAI frontend against the official openai Python SDK.

Run against cmd/fakeserver (its canned transcript and failure models are
assumed). Start it with -max-upload 100000; put any audio as a.wav and a
file starting with "corrupt" as corrupt.wav in the mounted directory:

    fakeserver -addr 127.0.0.1:18080 -max-upload 100000 &
    docker run --rm --network host -v "$PWD":/w python:3.12-slim sh -c \
        'pip install -q openai && python /w/sdk_check.py http://127.0.0.1:18080/v1 /w/a.wav'
"""

import sys

import openai

# --real-gateway: run against the real gateway with deploy/e2e's fake worker
# instead of this fakeserver. Three checks assert tricks of the canned
# service (the "unauth" model, the worker's own error text, an exact model
# list); with the flag they are skipped or relaxed.
REAL = "--real-gateway" in sys.argv
args = [a for a in sys.argv[1:] if a != "--real-gateway"]
BASE, AUDIO = args[0], args[1]
TEXT = "Hello world. This is a test. Goodbye."
c = openai.OpenAI(base_url=BASE, api_key="sk-test", max_retries=0)
print("openai SDK", openai.__version__)


def f():
    return open(AUDIO, "rb")


def check(name, cond):
    if not cond:
        raise SystemExit("FAIL: " + name)
    print("ok:", name)


r = c.audio.transcriptions.create(model="whisper-1", file=f())
check("json", r.text == TEXT and r.usage.type == "duration" and r.usage.seconds == 4)

r = c.audio.transcriptions.create(model="whisper-1", file=f(), response_format="text")
check("text", isinstance(r, str) and r.strip() == TEXT)

r = c.audio.transcriptions.create(model="whisper-1", file=f(), response_format="srt")
check("srt", isinstance(r, str) and r.startswith("1\n00:00:00,000 --> 00:00:01,200\nHello world."))

r = c.audio.transcriptions.create(model="whisper-1", file=f(), response_format="vtt")
check("vtt", isinstance(r, str) and r.startswith("WEBVTT\n\n00:00:00.000 --> "))

r = c.audio.transcriptions.create(model="whisper-1", file=f(), response_format="verbose_json",
                                  language="en", temperature=0.2, prompt="test")
check("verbose_json", r.language == "english" and r.duration == 3.5 and len(r.segments) == 3
      and r.segments[1].avg_logprob == -0.3 and r.words is None and r.usage.seconds == 4)

r = c.audio.transcriptions.create(model="whisper-1", file=f(), response_format="verbose_json",
                                  timestamp_granularities=["word"])
check("verbose_json words only", len(r.words) == 7 and r.words[0].word == "Hello" and r.segments is None)

r = c.audio.transcriptions.create(model="whisper-1", file=f(), response_format="verbose_json",
                                  timestamp_granularities=["word", "segment"])
check("verbose_json words+segments", len(r.words) == 7 and len(r.segments) == 3)

r = c.audio.transcriptions.create(model="whisper-1", file=f(), chunking_strategy="auto",
                                  include=["logprobs"])
check("gpt-only fields ignored", r.text == TEXT)

r = c.audio.transcriptions.create(model="whisper-1", file=f(),
                                  chunking_strategy={"type": "server_vad", "threshold": 0.5})
check("chunking_strategy object ignored", r.text == TEXT)

events = list(c.audio.transcriptions.create(model="whisper-1", file=f(), stream=True))
deltas = [e.delta for e in events if e.type == "transcript.text.delta"]
done = events[-1]
check("stream", len(deltas) == 3 and "".join(deltas) == TEXT and done.type == "transcript.text.done"
      and done.text == TEXT)

try:
    for e in c.audio.transcriptions.create(model="midfail", file=f(), stream=True):
        pass
    check("stream mid error raises", False)
except openai.APIError as e:
    check("stream mid error raises APIError", REAL or "went away" in str(e))

r = c.audio.translations.create(model="whisper-1", file=f())
check("translation json", r.text == TEXT)

r = c.audio.translations.create(model="whisper-1", file=f(), response_format="verbose_json")
check("translation verbose_json", r.language == "english" and len(r.segments) == 3)

r = c.audio.translations.create(model="whisper-1", file=f(), response_format="srt")
check("translation srt", isinstance(r, str) and r.startswith("1\n"))

ms = c.models.list()
ids = [m.id for m in ms.data]
if REAL:
    # The fake worker also serves its failure models; default first, alias last.
    check("models.list", ids[0] == "tiny" and ids[-1] == "whisper-1" and "base" in ids and ms.data[0].object == "model")
else:
    check("models.list", ids == ["tiny", "base", "whisper-1"] and ms.data[0].object == "model")

m = c.models.retrieve("whisper-1")
check("models.retrieve alias", m.id == "whisper-1" and m.owned_by == "whisper-hub")

m = c.models.retrieve("base")
check("models.retrieve id", m.id == "base" and m.object == "model")


def expect(name, exc, status, fn):
    try:
        fn()
    except exc as e:
        check(f"{name} -> {exc.__name__} {e.status_code} {e.code}", e.status_code == status
              and isinstance(e.body, dict) and "message" in e.body)
        return
    raise SystemExit("FAIL: no exception for " + name)


expect("models.retrieve missing", openai.NotFoundError, 404, lambda: c.models.retrieve("nope"))
expect("model not found", openai.NotFoundError, 404,
       lambda: c.audio.transcriptions.create(model="missing", file=f()))
expect("forbidden", openai.PermissionDeniedError, 403,
       lambda: c.audio.transcriptions.create(model="forbidden", file=f()))
if not REAL:
    expect("unauthenticated", openai.AuthenticationError, 401,
           lambda: c.audio.transcriptions.create(model="unauth", file=f()))
else:
    expect("unauthenticated", openai.AuthenticationError, 401,
           lambda: openai.OpenAI(base_url=BASE, api_key="sk-wrong", max_retries=0).audio.transcriptions.create(
               model="whisper-1", file=f()))
expect("unavailable", openai.InternalServerError, 503,
       lambda: c.audio.transcriptions.create(model="busy", file=f()))
expect("internal", openai.InternalServerError, 500,
       lambda: c.audio.transcriptions.create(model="boom", file=f()))
expect("bad audio", openai.BadRequestError, 400,
       lambda: c.audio.transcriptions.create(model="whisper-1", file=open(AUDIO.replace("a.wav", "corrupt.wav"), "rb")))
expect("bad response_format", openai.BadRequestError, 400,
       lambda: c.audio.transcriptions.create(model="whisper-1", file=f(), response_format="xml"))
expect("diarized_json", openai.BadRequestError, 400,
       lambda: c.audio.transcriptions.create(model="whisper-1", file=f(), response_format="diarized_json"))
expect("bad language", openai.BadRequestError, 400,
       lambda: c.audio.transcriptions.create(model="whisper-1", file=f(), language="xx"))
expect("bad temperature", openai.BadRequestError, 400,
       lambda: c.audio.transcriptions.create(model="whisper-1", file=f(), temperature=3))
expect("too large", openai.APIStatusError, 413,
       lambda: c.audio.transcriptions.create(model="whisper-1", file=("big.wav", b"\0" * 200000)))
print("ALL OK")
