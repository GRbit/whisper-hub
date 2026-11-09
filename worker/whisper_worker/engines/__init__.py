"""Engine factory. Backends are imported lazily: each image installs only its own libraries."""

from ..config import Config
from .base import Engine


def cuda_probe(engine: str):
    """cuda_probe returns the CUDA availability check matching upstream for engine."""
    if engine == "fake":
        return lambda: False
    if engine == "faster_whisper":
        from .faster_whisper import cuda_available

        return cuda_available

    def torch_cuda() -> bool:
        import torch

        return torch.cuda.is_available()

    return torch_cuda


def create(cfg: Config) -> Engine:
    if cfg.engine == "fake":
        from .fake import FakeEngine

        return FakeEngine(cfg)
    if cfg.engine == "faster_whisper":
        from .faster_whisper import FasterWhisperEngine

        return FasterWhisperEngine(cfg)
    if cfg.engine == "whisperx":
        from .whisperx import WhisperXEngine

        return WhisperXEngine(cfg)
    if cfg.engine == "openai_whisper":
        from .openai_whisper import OpenAIWhisperEngine

        return OpenAIWhisperEngine(cfg)
    raise ValueError(f"Unsupported ASR engine: {cfg.engine}")
