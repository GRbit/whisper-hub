"""openai-whisper backend.

Engine calls mirror upstream app/asr_models/openai_whisper_engine.py
(whisper-asr-webservice, MIT License) so legacy output stays byte-identical.

whisper.transcribe() has no callback, but it updates a (disabled) tqdm
progress bar after every 30 s window. We swap in a tqdm subclass that, for
the job running on the current thread, checks for cancellation and streams
the segments decoded so far. If a future whisper release changes those
internals the hook degrades to emitting everything at the end.
"""

import logging
import sys
import threading
import types
from typing import Any, Tuple

from ..errors import JobCancelled
from ..model_resolver import is_hf_reference
from .base import Done, Engine, JobContext, Options, empty_torch_cache, make_segment, make_word

logger = logging.getLogger("whisper_worker")

# Same options upstream passes to the whisper writers for every format.
WRITER_OPTIONS = {"max_line_width": 1000, "max_line_count": 10, "highlight_words": False}

_current = threading.local()
_hook_lock = threading.Lock()
_hook_installed = False


def _install_progress_hook() -> None:
    global _hook_installed
    with _hook_lock:
        if _hook_installed:
            return
        import importlib

        import tqdm

        # Not `import whisper.transcribe`: whisper/__init__ shadows that submodule with the function.
        wt = importlib.import_module("whisper.transcribe")

        class _ProgressHook(tqdm.tqdm):
            def update(self, n=1):
                r = super().update(n)
                stream = getattr(_current, "stream", None)
                if stream is not None:
                    stream.progress(sys._getframe(1))
                return r

        wt.tqdm = types.SimpleNamespace(tqdm=_ProgressHook)
        _hook_installed = True


def neutral(seg: dict) -> dict:
    words = None
    if seg.get("words") is not None:
        words = [make_word(w.get("word", ""), w.get("start"), w.get("end"), w.get("probability")) for w in seg["words"]]
    return make_segment(
        id=seg.get("id", 0),
        seek=seg.get("seek", 0),
        start=seg.get("start"),
        end=seg.get("end"),
        text=seg.get("text", ""),
        tokens=seg.get("tokens"),
        temperature=seg.get("temperature"),
        avg_logprob=seg.get("avg_logprob"),
        compression_ratio=seg.get("compression_ratio"),
        no_speech_prob=seg.get("no_speech_prob"),
        words=words,
    )


class _Stream:
    def __init__(self, ctx: JobContext, duration: float):
        self.ctx = ctx
        self.duration = duration
        self.info_sent = False
        self.sent = 0

    def send_info(self, language):
        if not self.info_sent:
            self.ctx.info(language, None, self.duration)
            self.info_sent = True

    def send_segments(self, segments):
        for seg in segments[self.sent:]:
            self.ctx.segment(neutral(seg))
        self.sent = max(self.sent, len(segments))

    def progress(self, frame):
        if self.ctx.cancelled():
            raise JobCancelled()
        try:
            local = frame.f_locals
            segments = local.get("all_segments")
            language = local.get("language")
        except Exception:  # internals changed: fall back to emitting at the end
            logger.debug("openai_whisper progress hook could not read transcribe state", exc_info=True)
            return
        if not isinstance(segments, list) or not isinstance(language, str):
            return
        self.send_info(language)
        self.send_segments(segments)


class OpenAIWhisperEngine(Engine):
    name = "openai_whisper"
    capabilities = {"word_timestamps": True, "vad_filter": False, "diarize": False, "translate": True}

    def version(self) -> str:
        import torch
        import whisper

        return f"openai-whisper {whisper.__version__}, torch {torch.__version__}"

    def load(self, model_id: str) -> Any:
        import torch
        import whisper

        if is_hf_reference(model_id):
            raise ValueError(
                f"ASR_MODEL={model_id!r} is a Hugging Face reference, which openai_whisper cannot load. "
                "Use ASR_ENGINE=faster_whisper or whisperx."
            )
        _install_progress_hook()
        # Upstream picks CUDA by availability and ignores ASR_DEVICE for this engine.
        if torch.cuda.is_available():
            return whisper.load_model(name=model_id, download_root=self.cfg.model_path).cuda()
        return whisper.load_model(name=model_id, download_root=self.cfg.model_path)

    def unload(self, handle: Any) -> None:
        super().unload(handle)
        empty_torch_cache()

    def transcribe(self, handle: Any, audio: Any, opts: Options, ctx: JobContext) -> Done:
        from whisper.utils import ResultWriter, WriteJSON, WriteSRT, WriteTSV, WriteTXT, WriteVTT

        options_dict = {"task": opts.task}
        if opts.language:
            options_dict["language"] = opts.language
        if opts.prompt:
            options_dict["initial_prompt"] = opts.prompt
        if opts.word_timestamps:
            options_dict["word_timestamps"] = opts.word_timestamps
        if opts.temperature is not None:
            options_dict["temperature"] = opts.temperature
        logger.debug("openai_whisper transcribe: %s", {k: v for k, v in options_dict.items() if k != "initial_prompt"})

        stream = _Stream(ctx, audio.shape[0] / 16000)
        _current.stream = stream
        try:
            result = handle.transcribe(audio, **options_dict)
        finally:
            _current.stream = None
        ctx.check_cancelled()
        stream.send_info(result["language"])
        stream.send_segments(result["segments"])

        rendered = None
        if opts.legacy_output:
            from io import StringIO

            file = StringIO()
            output = opts.legacy_output
            if output == "srt":
                WriteSRT(ResultWriter).write_result(result, file=file, options=WRITER_OPTIONS)
            elif output == "vtt":
                WriteVTT(ResultWriter).write_result(result, file=file, options=WRITER_OPTIONS)
            elif output == "tsv":
                WriteTSV(ResultWriter).write_result(result, file=file, options=WRITER_OPTIONS)
            elif output == "json":
                WriteJSON(ResultWriter).write_result(result, file=file, options=WRITER_OPTIONS)
            else:
                WriteTXT(ResultWriter).write_result(result, file=file, options=WRITER_OPTIONS)
            rendered = file.getvalue()
        return Done(text=result["text"], language=result["language"], rendered=rendered)

    def detect_language(self, handle: Any, audio: Any) -> Tuple[str, float]:
        import whisper

        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, handle.dims.n_mels).to(handle.device)
        _, probs = handle.detect_language(mel)
        detected_lang_code = max(probs, key=probs.get)
        # Upstream quirk kept for legacy fidelity: max(probs) is the alphabetically
        # last code ("zh"), so the reported confidence is P(zh), not P(detected).
        return detected_lang_code, float(probs[max(probs)])
