"""Deterministic engine for gateway tests: no model, no inference library.

Behavior (documented in worker/README.md, keep both in sync):

- Segment i (0-based) has id i+1, seek i*200, start 2*i, end 2*i+2 and the
  text TEXTS[i % 3]. WORKER_FAKE_SEGMENTS (default 3) sets the count.
- Language: options.language when given (probability 1.0), otherwise
  WORKER_FAKE_LANGUAGE (default "en", probability 0.99). task is ignored.
- word_timestamps: each segment gets words split on spaces, spread evenly
  over the segment, probability 0.9. Otherwise words is null.
- legacy_output renders exactly like upstream faster_whisper would for the
  same segments.
- WORKER_FAKE_DELAY seconds are waited before each segment (cancellable).
- A prompt starting with "fake:" holds directives instead of a prompt,
  separated by commas or spaces:
    delay=<seconds>   override WORKER_FAKE_DELAY
    segments=<n>      override WORKER_FAKE_SEGMENTS
    fail=bad_request|model_not_found|unavailable|internal
                      fail before any output (HTTP error)
    fail_after=<n>    emit n segments, then an error event (code internal)
- /v1/detect-language returns WORKER_FAKE_LANGUAGE with confidence 0.99.
"""

import logging
from typing import Any, Tuple

from .. import legacy_writers
from ..audio import duration
from ..errors import BadRequest, Internal, ModelNotFound, Unavailable
from .base import Done, Engine, JobContext, Options, to_neutral

logger = logging.getLogger("whisper_worker")

TEXTS = (
    " Hello from the fake engine.",
    " This is segment number two.",
    " The quick brown fox jumps over the lazy dog.",
)
SEGMENT_SECONDS = 2.0

FAILURES = {
    "bad_request": BadRequest,
    "model_not_found": ModelNotFound,
    "unavailable": Unavailable,
    "internal": Internal,
}


class Directives:
    def __init__(self, delay: float, segments: int):
        self.delay = delay
        self.segments = segments
        self.fail = None
        self.fail_after = None


def parse_directives(prompt, delay: float, segments: int) -> Directives:
    d = Directives(delay, segments)
    if not prompt or not prompt.startswith("fake:"):
        return d
    for item in prompt[len("fake:"):].replace(",", " ").split():
        key, _, value = item.partition("=")
        try:
            if key == "delay":
                d.delay = max(0.0, float(value))
            elif key == "segments":
                d.segments = max(0, int(value))
            elif key == "fail":
                if value not in FAILURES:
                    raise ValueError(value)
                d.fail = value
            elif key == "fail_after":
                d.fail_after = max(0, int(value))
            else:
                raise ValueError(key)
        except ValueError:
            raise BadRequest(f"unknown fake directive {item!r}") from None
    return d


def fake_segments(n: int, word_timestamps: bool):
    """fake_segments returns faster_whisper look-alike Segment objects."""
    out = []
    for i in range(n):
        text = TEXTS[i % len(TEXTS)]
        start = SEGMENT_SECONDS * i
        end = start + SEGMENT_SECONDS
        words = None
        if word_timestamps:
            parts = text.split(" ")[1:]
            step = SEGMENT_SECONDS / len(parts)
            words = [
                legacy_writers.Word(
                    start=round(start + k * step, 3), end=round(start + (k + 1) * step, 3), word=" " + w, probability=0.9
                )
                for k, w in enumerate(parts)
            ]
        out.append(
            legacy_writers.Segment(
                id=i + 1,
                seek=i * 200,
                start=start,
                end=end,
                text=text,
                tokens=[50364 + i],
                avg_logprob=-0.25,
                compression_ratio=1.2,
                no_speech_prob=0.01,
                words=words,
                temperature=0.0,
            )
        )
    return out


class FakeEngine(Engine):
    name = "fake"
    capabilities = {"word_timestamps": True, "vad_filter": True, "diarize": False, "translate": True}

    def version(self) -> str:
        return "fake"

    def validate(self, opts: Options) -> None:
        d = parse_directives(opts.prompt, self.cfg.fake_delay, self.cfg.fake_segments)
        if d.fail:
            raise FAILURES[d.fail](f"fake failure requested: {d.fail}")

    def prepare_audio(self, pcm: bytes) -> Any:
        return pcm

    def load(self, model_id: str) -> Any:
        return {"model": model_id}

    def transcribe(self, handle: Any, audio: Any, opts: Options, ctx: JobContext) -> Done:
        d = parse_directives(opts.prompt, self.cfg.fake_delay, self.cfg.fake_segments)
        if opts.language:
            language, prob = opts.language, 1.0
        else:
            language, prob = self.cfg.fake_language, 0.99
        ctx.info(language, prob, duration(audio))
        segments = []
        text = ""
        for seg in fake_segments(d.segments, opts.word_timestamps):
            if d.fail_after is not None and len(segments) >= d.fail_after:
                raise Internal("fake failure requested after segments")
            # Event.wait returns early on cancellation, so a slow fake stream stops at once.
            if d.delay and ctx.cancel_event.wait(d.delay):
                ctx.check_cancelled()
            ctx.check_cancelled()
            segments.append(seg)
            text += seg.text
            ctx.segment(to_neutral(seg))
        if d.fail_after is not None and len(segments) >= d.fail_after:
            raise Internal("fake failure requested after segments")
        rendered = None
        if opts.legacy_output:
            rendered = legacy_writers.render({"language": language, "segments": segments, "text": text}, opts.legacy_output)
        return Done(text=text, language=language, rendered=rendered)

    def detect_language(self, handle: Any, audio: Any) -> Tuple[str, float]:
        return self.cfg.fake_language, 0.99
