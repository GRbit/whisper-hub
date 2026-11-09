"""Environment configuration.

Upstream variable names keep their whisper-asr-webservice semantics
(app/config.py there); settings added by whisper-hub use the WORKER_ prefix.
"""

import os
from dataclasses import dataclass, field
from typing import Callable, List, Mapping, Optional

ENGINES = ("faster_whisper", "whisperx", "openai_whisper", "fake")

QUANTIZATIONS = {
    "float32", "float16", "int8", "bfloat16", "int16", "int8_float32", "int8_float16", "int8_bfloat16"
}

LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


class ConfigError(ValueError):
    """ConfigError reports an invalid environment variable."""


@dataclass
class Config:
    """Config is the complete worker configuration."""

    engine: str = "openai_whisper"
    model: str = "base"
    model_path: str = ""
    device: str = "cpu"
    quantization: str = "int8"
    model_idle_timeout: int = 0
    hf_token: str = ""
    log_level: str = "INFO"
    subtitle_max_line_width: int = 1000
    subtitle_max_line_count: int = 2
    subtitle_highlight_words: bool = False

    host: str = "0.0.0.0"
    port: int = 9100
    # First entry is the default model.
    models: List[str] = field(default_factory=list)
    max_loaded_models: int = 1
    preload: bool = False

    fake_segments: int = 3
    fake_delay: float = 0.0
    fake_language: str = "en"

    @property
    def default_model(self) -> str:
        return self.models[0]


def _int(env: Mapping[str, str], name: str, default: int, minimum: Optional[int] = None) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name}={raw!r} is not an integer") from None
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name}={raw!r} must be >= {minimum}")
    return value


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{name}={raw!r} is not a number") from None
    if value < 0:
        raise ConfigError(f"{name}={raw!r} must be >= 0")
    return value


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{name}={raw!r} is not a boolean")


def from_env(env: Optional[Mapping[str, str]] = None, cuda_available: Callable[[], bool] = lambda: False) -> Config:
    """from_env builds a Config from environment variables.

    cuda_available is only consulted for the upstream defaults of ASR_DEVICE
    and ASR_QUANTIZATION; it is a callable so that the fake engine never has
    to import an inference library.
    """
    env = os.environ if env is None else env
    c = Config()

    c.engine = env.get("ASR_ENGINE", "openai_whisper")
    if c.engine not in ENGINES:
        raise ConfigError(f"Unsupported ASR engine: {c.engine}")

    c.hf_token = env.get("HF_TOKEN", "")
    c.model = env.get("ASR_MODEL", "base")
    c.model_path = env.get("ASR_MODEL_PATH", os.path.join(os.path.expanduser("~"), ".cache", "whisper"))

    cuda = None

    def has_cuda() -> bool:
        nonlocal cuda
        if cuda is None:
            cuda = bool(cuda_available())
        return cuda

    c.device = env.get("ASR_DEVICE") or ("cuda" if has_cuda() else "cpu")
    # Upstream bases the quantization default on CUDA presence, not on ASR_DEVICE.
    c.quantization = env.get("ASR_QUANTIZATION") or ("float32" if has_cuda() else "int8")
    if c.quantization not in QUANTIZATIONS:
        raise ConfigError(
            "Invalid MODEL_QUANTIZATION. Choose 'float32', 'float16', or 'int8' "
            "(or another CTranslate2 compute type: 'bfloat16', 'int16', 'int8_float32', 'int8_float16', 'int8_bfloat16')."
        )

    c.model_idle_timeout = _int(env, "MODEL_IDLE_TIMEOUT", 0)
    c.log_level = env.get("LOG_LEVEL", "INFO").upper()
    if c.log_level not in LOG_LEVELS:
        raise ConfigError(f"LOG_LEVEL={c.log_level!r}: choose one of {', '.join(LOG_LEVELS)}")

    c.subtitle_max_line_width = _int(env, "SUBTITLE_MAX_LINE_WIDTH", 1000)
    c.subtitle_max_line_count = _int(env, "SUBTITLE_MAX_LINE_COUNT", 2)
    # Upstream accepts only the literal "true" (any case); everything else is false.
    c.subtitle_highlight_words = env.get("SUBTITLE_HIGHLIGHT_WORDS", "false").lower() == "true"

    c.host = env.get("WORKER_HOST", "0.0.0.0")
    c.port = _int(env, "WORKER_PORT", 9100, minimum=1)
    raw_models = env.get("WORKER_MODELS", "")
    models = []
    for m in raw_models.split(","):
        m = m.strip()
        if m and m not in models:
            models.append(m)
    c.models = models or [c.model]
    c.max_loaded_models = _int(env, "WORKER_MAX_LOADED_MODELS", 1, minimum=1)
    c.preload = _bool(env, "WORKER_PRELOAD", False)

    c.fake_segments = _int(env, "WORKER_FAKE_SEGMENTS", 3, minimum=0)
    c.fake_delay = _float(env, "WORKER_FAKE_DELAY", 0.0)
    c.fake_language = env.get("WORKER_FAKE_LANGUAGE", "en")
    return c
