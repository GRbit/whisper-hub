"""Engine interface shared by all backends.

Engine methods are synchronous and run on a worker thread; they report
progress through a JobContext and must call ctx.check_cancelled() whenever
the library hands control back to them.
"""

import gc
import logging
import math
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config import Config
from ..errors import BadRequest, JobCancelled
from ..languages import LANGUAGES

logger = logging.getLogger("whisper_worker")

LEGACY_OUTPUTS = ("txt", "vtt", "srt", "tsv", "json")
TASKS = ("transcribe", "translate")


@dataclass
class Options:
    """Options are the parsed `options` part of POST /v1/transcribe."""

    model: Optional[str] = None
    task: str = "transcribe"
    language: Optional[str] = None
    prompt: Optional[str] = None
    temperature: Optional[float] = None
    vad_filter: bool = False
    word_timestamps: bool = False
    diarize: bool = False
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    legacy_output: Optional[str] = None


def _opt_str(raw: dict, key: str) -> Optional[str]:
    v = raw.get(key)
    if v is None:
        return None
    if not isinstance(v, str):
        raise BadRequest(f"options.{key} must be a string or null")
    # Upstream treats empty strings like absent values (`if language:`).
    return v or None


def _opt_bool(raw: dict, key: str) -> bool:
    v = raw.get(key)
    if v is None:
        return False
    if not isinstance(v, bool):
        raise BadRequest(f"options.{key} must be a boolean")
    return v


def _opt_int(raw: dict, key: str) -> Optional[int]:
    v = raw.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise BadRequest(f"options.{key} must be a non-negative integer or null")
    return v


def parse_options(raw: Any) -> Options:
    """parse_options validates the decoded options JSON. Unknown keys are ignored."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise BadRequest("options must be a JSON object")
    o = Options()
    o.model = _opt_str(raw, "model")
    o.task = _opt_str(raw, "task") or "transcribe"
    if o.task not in TASKS:
        raise BadRequest(f"options.task must be one of {', '.join(TASKS)}")
    o.language = _opt_str(raw, "language")
    if o.language is not None and o.language not in LANGUAGES:
        raise BadRequest(f"options.language {o.language!r} is not a Whisper language code")
    o.prompt = _opt_str(raw, "prompt")
    t = raw.get("temperature")
    if t is not None:
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
            raise BadRequest("options.temperature must be a non-negative number or null")
        o.temperature = float(t)
    o.vad_filter = _opt_bool(raw, "vad_filter")
    o.word_timestamps = _opt_bool(raw, "word_timestamps")
    o.diarize = _opt_bool(raw, "diarize")
    o.min_speakers = _opt_int(raw, "min_speakers")
    o.max_speakers = _opt_int(raw, "max_speakers")
    o.legacy_output = _opt_str(raw, "legacy_output")
    if o.legacy_output is not None and o.legacy_output not in LEGACY_OUTPUTS:
        raise BadRequest(f"options.legacy_output must be one of {', '.join(LEGACY_OUTPUTS)} or null")
    return o


@dataclass
class Done:
    """Done is the final outcome of a transcription."""

    text: str
    language: str
    rendered: Optional[str] = None


class JobContext:
    """JobContext connects an engine running on a thread to the HTTP stream."""

    def __init__(self, emit: Callable[[dict], None], cancel_event: threading.Event, model: str, engine: str):
        self._emit = emit
        self.cancel_event = cancel_event
        self.model = model
        self.engine = engine

    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise JobCancelled()

    def info(self, language: str, language_probability: Optional[float], duration: float) -> None:
        self._emit(
            {
                "type": "info",
                "model": self.model,
                "engine": self.engine,
                "language": language,
                "language_probability": _num(language_probability),
                "duration": _num(duration),
            }
        )

    def segment(self, segment: dict) -> None:
        self._emit({"type": "segment", "segment": segment})


def _num(v) -> Optional[float]:
    """_num makes engine numbers safe for Go's JSON parser: numpy scalars become floats and
    NaN/Infinity (whisperx alignment can produce them) become 0. Absent values stay null."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else 0.0


def make_word(word: str, start, end, probability, speaker=None) -> dict:
    return {
        "word": word,
        "start": _num(start),
        "end": _num(end),
        "probability": _num(probability),
        "speaker": speaker,
    }


def make_segment(
    *,
    id: int,
    seek: int = 0,
    start,
    end,
    text: str,
    tokens: Optional[List[int]] = None,
    temperature=None,
    avg_logprob=None,
    compression_ratio=None,
    no_speech_prob=None,
    words: Optional[List[dict]] = None,
    speaker: Optional[str] = None,
) -> dict:
    """make_segment builds the engine-neutral segment of the Worker API."""
    return {
        "id": int(id),
        "seek": int(seek),
        "start": _num(start),
        "end": _num(end),
        "text": text,
        "tokens": [int(t) for t in (tokens or [])],
        "temperature": _num(temperature),
        "avg_logprob": _num(avg_logprob),
        "compression_ratio": _num(compression_ratio),
        "no_speech_prob": _num(no_speech_prob),
        "words": words,
        "speaker": speaker,
    }


def to_neutral(seg) -> dict:
    """to_neutral converts a faster_whisper style Segment object to the Worker API shape."""
    words = None
    if seg.words is not None:
        words = [make_word(w.word, w.start, w.end, w.probability) for w in seg.words]
    return make_segment(
        id=seg.id,
        seek=seg.seek,
        start=seg.start,
        end=seg.end,
        text=seg.text,
        tokens=seg.tokens,
        temperature=seg.temperature,
        avg_logprob=seg.avg_logprob,
        compression_ratio=seg.compression_ratio,
        no_speech_prob=seg.no_speech_prob,
        words=words,
    )


class Engine:
    """Engine is one speech recognition backend; see module docstring."""

    name = ""
    capabilities: Dict[str, bool] = {"word_timestamps": False, "vad_filter": False, "diarize": False, "translate": True}

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def version(self) -> str:
        return ""

    def validate(self, opts: Options) -> None:
        """validate rejects options the engine cannot honor; called before queueing."""

    def prepare_audio(self, pcm: bytes) -> Any:
        from ..audio import to_float32

        return to_float32(pcm)

    def load(self, model_id: str) -> Any:
        raise NotImplementedError

    def unload(self, handle: Any) -> None:
        del handle
        gc.collect()

    def transcribe(self, handle: Any, audio: Any, opts: Options, ctx: JobContext) -> Done:
        raise NotImplementedError

    def detect_language(self, handle: Any, audio: Any) -> Tuple[str, float]:
        raise NotImplementedError


def empty_torch_cache() -> None:
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # torch missing or no CUDA: nothing to free
        logger.debug("torch.cuda.empty_cache skipped", exc_info=True)
