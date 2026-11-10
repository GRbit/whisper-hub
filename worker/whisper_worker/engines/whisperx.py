"""WhisperX backend: batched transcription, forced alignment, optional diarization.

Engine calls mirror upstream app/asr_models/mbain_whisperx_engine.py
(whisper-asr-webservice, MIT License) so legacy output matches upstream's.
WhisperX returns everything at once. Pausing is possible only between the
transcribe, align and diarize stages; cancellation is also checked between
inference batches of transcribe and between segments of align, through
hooks that raise JobCancelled (see install_batch_hook and CancelableModel).
"""

import collections
import contextlib
import dataclasses
import logging
import threading
import typing
from io import StringIO
from typing import Any, Tuple

from ..errors import JobCancelled
from ..model_resolver import resolve_model
from .base import Done, Engine, JobContext, Options, empty_torch_cache, make_segment, make_word

logger = logging.getLogger("whisper_worker")


def _vad_checkpoint_globals() -> list:
    import omegaconf
    import torch
    from pyannote.audio.core.model import Introspection
    from pyannote.audio.core.task import Problem, Resolution, Specifications

    return [
        omegaconf.listconfig.ListConfig, omegaconf.base.ContainerMetadata, omegaconf.nodes.AnyNode,
        omegaconf.base.Metadata, typing.Any, list, dict, int, collections.defaultdict,
        torch.torch_version.TorchVersion, Introspection, Specifications, Problem, Resolution,
    ]


_job = threading.local()
_warned: set = set()


class _Scope:
    def __init__(self, ctx):
        self.ctx = ctx
        self.checks: dict = {}

    def count(self, what: str) -> int:
        return self.checks.get(what, 0)

    def expect(self, what: str) -> None:
        """expect warns once per process when a hook of a stage that did run never fired."""
        if self.count(what) == 0 and what not in _warned:
            _warned.add(what)
            logger.warning("whisperx %s cancel hook never ran: this whisperx version calls its model differently, "
                           "cancellation falls back to stage boundaries", what)


@contextlib.contextmanager
def cancel_scope(ctx):
    """cancel_scope makes the hooks on this thread check ctx; outside of it they do nothing."""
    scope = _Scope(ctx)
    _job.scope = scope
    try:
        yield scope
    finally:
        _job.scope = None


def _check(what: str) -> None:
    scope = getattr(_job, "scope", None)
    if scope is None:
        return
    scope.checks[what] = scope.count(what) + 1
    if scope.ctx.cancelled():
        raise JobCancelled()


def install_batch_hook(pipeline) -> bool:
    """install_batch_hook makes pipeline._forward, which transformers calls once per inference
    batch, raise JobCancelled for a canceled job first. Only cancellation: the pipeline's tokenizer
    and options belong to the call in progress, so no other job may run in between. Returns False,
    with a warning, for a pipeline of an unexpected shape, which keeps cancellation at stage
    boundaries."""
    forward = getattr(pipeline, "_forward", None)
    if not callable(forward):
        logger.warning("whisperx pipeline has no _forward: cancellation only at stage boundaries")
        return False
    if getattr(forward, "_whisper_hub_hook", False):
        return True

    def hooked(model_inputs, *args, **kwargs):
        _check("transcribe batch")
        return forward(model_inputs, *args, **kwargs)

    hooked._whisper_hub_hook = True
    pipeline._forward = hooked
    return True


def install_vad_hook(pipeline) -> bool:
    """install_vad_hook does the same for voice activity detection, which runs over the whole
    file before the first batch: pyannote's Inference calls its infer once per batch of chunks.
    Returns False, with a warning, when the VAD is not that shape (another VAD method or
    library version)."""
    seg = getattr(getattr(getattr(pipeline, "vad_model", None), "vad_pipeline", None), "_segmentation", None)
    infer = getattr(seg, "infer", None)
    if not callable(infer):
        logger.warning("whisperx VAD is not the expected pyannote shape: cancellation during voice activity "
                       "detection only at stage boundaries")
        return False
    if getattr(infer, "_whisper_hub_hook", False):
        return True

    def hooked(chunks, *args, **kwargs):
        _check("vad batch")
        return infer(chunks, *args, **kwargs)

    hooked._whisper_hub_hook = True
    seg.infer = hooked
    return True


class CancelableModel:
    """CancelableModel wraps an align model, which whisperx.align calls once per segment, so
    that a canceled job stops before the next segment. Everything else is delegated."""

    def __init__(self, model):
        self._model = model

    def __call__(self, *args, **kwargs):
        _check("align segment")
        return self._model(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)


def neutral(i: int, seg: dict) -> dict:
    words = None
    if seg.get("words") is not None:
        words = [
            make_word(w.get("word", ""), w.get("start"), w.get("end"), w.get("score"), w.get("speaker"))
            for w in seg["words"]
        ]
    return make_segment(
        id=i,
        start=seg.get("start"),
        end=seg.get("end"),
        text=seg.get("text", ""),
        words=words,
        speaker=seg.get("speaker"),
    )


class WhisperXEngine(Engine):
    name = "whisperx"
    # Only between stages: the pipeline's tokenizer and options belong to the
    # call in progress during transcribe, but are reset when it returns.
    pausable = True

    def __init__(self, cfg):
        super().__init__(cfg)
        # Alignment always yields word timings; VAD is always on and not switchable,
        # so vad_filter is not advertised (upstream ignores it for whisperx too).
        self.capabilities = {
            "word_timestamps": True,
            "vad_filter": False,
            "diarize": bool(cfg.hf_token),
            "translate": True,
        }

    def version(self) -> str:
        import importlib.metadata

        import torch

        return f"whisperx {importlib.metadata.version('whisperx')}, torch {torch.__version__}"

    def load(self, model_id: str) -> Any:
        import torch
        import whisperx

        cfg = self.cfg
        handle = {"whisperx": None, "diarize_model": None, "align_model": {}}
        # torch 2.6+ loads with weights_only=True, which rejects the pyannote VAD checkpoint
        # bundled in the whisperx wheel; the reference image fails to start for this reason.
        # Allowlist exactly the classes that checkpoint needs, only while loading it.
        with torch.serialization.safe_globals(_vad_checkpoint_globals()):
            handle["whisperx"] = whisperx.load_model(
                resolve_model(model_id, cfg.model_path, cfg.quantization, cfg.hf_token or None),
                device=cfg.device,
                compute_type=cfg.quantization,
                asr_options={"without_timestamps": False},
            )
        install_batch_hook(handle["whisperx"])
        install_vad_hook(handle["whisperx"])
        if cfg.hf_token != "":
            from whisperx.diarize import DiarizationPipeline

            handle["diarize_model"] = DiarizationPipeline(use_auth_token=cfg.hf_token, device=cfg.device)
        return handle

    def unload(self, handle: Any) -> None:
        handle.clear()
        super().unload(handle)
        empty_torch_cache()

    def transcribe(self, handle: Any, audio: Any, opts: Options, ctx: JobContext) -> Done:
        import whisperx
        from whisperx.utils import ResultWriter, SubtitlesWriter, WriteJSON, WriteSRT, WriteTSV, WriteTXT, WriteVTT

        cfg = self.cfg
        pipeline = handle["whisperx"]
        options_dict = {"task": opts.task}
        if opts.language:
            options_dict["language"] = opts.language

        # Before the options swap below, which must not be seen by another job.
        ctx.check_cancelled()
        # Upstream passes initial_prompt as a transcribe() kwarg, which this whisperx
        # version rejects with TypeError (HTTP 500). The pipeline takes it through its
        # TranscriptionOptions instead; temperature goes the same way.
        saved_options = pipeline.options
        # A cancel inside transcribe skips the pipeline's own reset of its tokenizer, which
        # would otherwise carry this job's language into the next job.
        saved_tokenizer = getattr(pipeline, "tokenizer", None)
        overrides = {}
        if opts.prompt:
            overrides["initial_prompt"] = opts.prompt
        if opts.temperature is not None:
            overrides["temperatures"] = [opts.temperature]
        if overrides:
            pipeline.options = dataclasses.replace(saved_options, **overrides)
        logger.debug("whisperx transcribe: %s overrides=%s", options_dict, sorted(overrides))
        try:
            with cancel_scope(ctx) as scope:
                result = pipeline.transcribe(audio, **options_dict)
            scope.expect("vad batch")
            if result.get("segments"):
                scope.expect("transcribe batch")
        except BaseException:
            if hasattr(pipeline, "tokenizer"):
                pipeline.tokenizer = saved_tokenizer
            raise
        finally:
            pipeline.options = saved_options
        language = result["language"]
        ctx.check_cancelled()

        # Cached per language; many languages may exhaust memory, as upstream notes.
        if result["language"] not in handle["align_model"]:
            handle["align_model"][result["language"]] = whisperx.load_align_model(
                language_code=result["language"], device=cfg.device
            )
        model_x, metadata = handle["align_model"][result["language"]]
        with cancel_scope(ctx) as scope:
            result = whisperx.align(result["segments"], CancelableModel(model_x), metadata, audio, cfg.device,
                                    return_char_alignments=False)
        if result.get("segments"):
            scope.expect("align segment")
        ctx.check_cancelled()

        if opts.diarize and cfg.hf_token != "":
            # Positional on purpose, as upstream: DiarizationPipeline.__call__ is
            # (audio, num_speakers, min_speakers, max_speakers), so min_speakers lands in
            # num_speakers and max_speakers in min_speakers. Kept for legacy fidelity.
            diarize_segments = handle["diarize_model"](audio, opts.min_speakers, opts.max_speakers)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            ctx.check_cancelled()
        result["language"] = language

        ctx.info(language, None, audio.shape[0] / 16000)
        for i, seg in enumerate(result["segments"]):
            ctx.segment(neutral(i, seg))
        text = "".join(seg.get("text", "") for seg in result["segments"])

        rendered = None
        if opts.legacy_output:
            default_options = {
                "max_line_width": cfg.subtitle_max_line_width,
                "max_line_count": cfg.subtitle_max_line_count,
                "highlight_words": cfg.subtitle_highlight_words,
            }
            file = StringIO()
            output = opts.legacy_output
            if output == "srt":
                WriteSRT(SubtitlesWriter).write_result(result, file=file, options=default_options)
            elif output == "vtt":
                WriteVTT(SubtitlesWriter).write_result(result, file=file, options=default_options)
            elif output == "tsv":
                WriteTSV(ResultWriter).write_result(result, file=file, options=default_options)
            elif output == "json":
                WriteJSON(ResultWriter).write_result(result, file=file, options=default_options)
            else:
                WriteTXT(ResultWriter).write_result(result, file=file, options=default_options)
            rendered = file.getvalue()
        return Done(text=text, language=language, rendered=rendered)

    def detect_language(self, handle: Any, audio: Any) -> Tuple[str, float]:
        # faster-whisper's detect_language reads only the first 30 s window.
        results = handle["whisperx"].model.detect_language(audio)
        return results[0], round(float(results[1]), 2)
