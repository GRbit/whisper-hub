"""faster-whisper backend. Streams segments as the library yields them.

Engine calls mirror upstream app/asr_models/faster_whisper_engine.py
(whisper-asr-webservice, MIT License) so legacy output stays byte-identical.
"""

import logging
from typing import Any, Tuple

from .. import legacy_writers
from ..audio import pad_or_trim
from ..model_resolver import resolve_model
from .base import Done, Engine, JobContext, Options, to_neutral

logger = logging.getLogger("whisper_worker")


def cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


class FasterWhisperEngine(Engine):
    name = "faster_whisper"
    capabilities = {"word_timestamps": True, "vad_filter": True, "diarize": False, "translate": True}

    def version(self) -> str:
        import ctranslate2
        import faster_whisper

        return f"faster-whisper {faster_whisper.__version__}, ctranslate2 {ctranslate2.__version__}"

    def load(self, model_id: str) -> Any:
        from faster_whisper import WhisperModel

        cfg = self.cfg
        return WhisperModel(
            model_size_or_path=resolve_model(model_id, cfg.model_path, cfg.quantization, cfg.hf_token or None),
            device=cfg.device,
            compute_type=cfg.quantization,
            download_root=cfg.model_path,
        )

    def transcribe(self, handle: Any, audio: Any, opts: Options, ctx: JobContext) -> Done:
        # Only set options are passed, exactly like upstream, so library defaults apply otherwise.
        options_dict = {"task": opts.task}
        if opts.language:
            options_dict["language"] = opts.language
        if opts.prompt:
            options_dict["initial_prompt"] = opts.prompt
        if opts.vad_filter:
            options_dict["vad_filter"] = True
        if opts.word_timestamps:
            options_dict["word_timestamps"] = True
        if opts.temperature is not None:
            # Not an upstream /asr option (it never sends one); used by the OpenAI frontend.
            options_dict["temperature"] = opts.temperature
        logger.debug("faster_whisper transcribe: beam_size=5 %s", {k: v for k, v in options_dict.items() if k != "initial_prompt"})
        segment_generator, info = handle.transcribe(audio, beam_size=5, **options_dict)
        language = options_dict.get("language", info.language)
        ctx.info(language, info.language_probability, info.duration)

        segments = []
        text = ""
        try:
            for segment in segment_generator:
                segments.append(segment)
                text = text + segment.text
                ctx.segment(to_neutral(segment))
                # The generator decodes the next window on demand, so stopping here stops the work.
                ctx.check_cancelled()
        finally:
            segment_generator.close()

        rendered = None
        if opts.legacy_output:
            rendered = legacy_writers.render({"language": language, "segments": segments, "text": text}, opts.legacy_output)
        return Done(text=text, language=language, rendered=rendered)

    def detect_language(self, handle: Any, audio: Any) -> Tuple[str, float]:
        # Upstream: pad/trim to 30 s and read the language from transcribe info; the
        # segment generator is never consumed, so no decoding happens.
        segments, info = handle.transcribe(pad_or_trim(audio), beam_size=5)
        return info.language, info.language_probability
