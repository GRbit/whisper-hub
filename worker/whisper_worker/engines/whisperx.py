"""WhisperX backend: batched transcription, forced alignment, optional diarization.

Engine calls mirror upstream app/asr_models/mbain_whisperx_engine.py
(whisper-asr-webservice, MIT License) so legacy output matches upstream's.
WhisperX returns everything at once; cancellation is checked between the
transcribe, align and diarize stages.
"""

import dataclasses
import collections
import logging
import typing
from io import StringIO
from typing import Any, Tuple

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

        # Upstream passes initial_prompt as a transcribe() kwarg, which this whisperx
        # version rejects with TypeError (HTTP 500). The pipeline takes it through its
        # TranscriptionOptions instead; temperature goes the same way.
        saved_options = pipeline.options
        overrides = {}
        if opts.prompt:
            overrides["initial_prompt"] = opts.prompt
        if opts.temperature is not None:
            overrides["temperatures"] = [opts.temperature]
        if overrides:
            pipeline.options = dataclasses.replace(saved_options, **overrides)
        logger.debug("whisperx transcribe: %s overrides=%s", options_dict, sorted(overrides))
        try:
            result = pipeline.transcribe(audio, **options_dict)
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
        result = whisperx.align(result["segments"], model_x, metadata, audio, cfg.device, return_char_alignments=False)
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
