"""Environment configuration.

Every setting has a WHISPER_HUB_ name: WHISPER_HUB_ENGINE and
WHISPER_HUB_ENGINE_* for the engine worker (the gateway's configuration has
no "engine" section, so these never collide with its derived variables),
and WHISPER_HUB_LOG_LEVEL, shared with the gateway's log.level. The names
of whisper-asr-webservice (app/config.py there) are accepted as aliases
with their upstream semantics; the WHISPER_HUB_ name wins when both are set.
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


# ALIASES maps each WHISPER_HUB_ name to the whisper-asr-webservice name a
# drop-in deployment may still use.
ALIASES = {
    "WHISPER_HUB_ENGINE": "ASR_ENGINE",
    "WHISPER_HUB_ENGINE_MODEL": "ASR_MODEL",
    "WHISPER_HUB_ENGINE_MODEL_PATH": "ASR_MODEL_PATH",
    "WHISPER_HUB_ENGINE_DEVICE": "ASR_DEVICE",
    "WHISPER_HUB_ENGINE_QUANTIZATION": "ASR_QUANTIZATION",
    "WHISPER_HUB_ENGINE_MODEL_IDLE_TIMEOUT": "MODEL_IDLE_TIMEOUT",
    "WHISPER_HUB_ENGINE_SUBTITLE_MAX_LINE_WIDTH": "SUBTITLE_MAX_LINE_WIDTH",
    "WHISPER_HUB_ENGINE_SUBTITLE_MAX_LINE_COUNT": "SUBTITLE_MAX_LINE_COUNT",
    "WHISPER_HUB_ENGINE_SUBTITLE_HIGHLIGHT_WORDS": "SUBTITLE_HIGHLIGHT_WORDS",
    "WHISPER_HUB_LOG_LEVEL": "LOG_LEVEL",
}

# The gateway's log.level spellings, so one WHISPER_HUB_LOG_LEVEL serves both.
_LEVEL_SPELLINGS = {"WARN": "WARNING", "FATAL": "CRITICAL"}


def lookup(env: Mapping[str, str], name: str) -> "tuple[str, Optional[str]]":
    """lookup returns the variable that sets name and its value.

    An empty value counts as unset, so an empty WHISPER_HUB_ name falls
    through to its alias and then to the default.
    """
    for key in (name, ALIASES.get(name)):
        if key and env.get(key):
            return key, env[key]
    return name, None


def _int(env: Mapping[str, str], name: str, default: int, minimum: Optional[int] = None) -> int:
    key, raw = lookup(env, name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{key}={raw!r} is not an integer") from None
    if minimum is not None and value < minimum:
        raise ConfigError(f"{key}={raw!r} must be >= {minimum}")
    return value


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    key, raw = lookup(env, name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{key}={raw!r} is not a number") from None
    if value < 0:
        raise ConfigError(f"{key}={raw!r} must be >= 0")
    return value


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    key, raw = lookup(env, name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{key}={raw!r} is not a boolean")


def _str(env: Mapping[str, str], name: str, default: str) -> str:
    return lookup(env, name)[1] or default


def from_env(env: Optional[Mapping[str, str]] = None, cuda_available: Callable[[], bool] = lambda: False) -> Config:
    """from_env builds a Config from environment variables.

    cuda_available is only consulted for the upstream defaults of the device
    and the quantization; it is a callable so that the fake engine never has
    to import an inference library.
    """
    env = os.environ if env is None else env
    c = Config()

    c.engine = _str(env, "WHISPER_HUB_ENGINE", "openai_whisper")
    if c.engine not in ENGINES:
        raise ConfigError(f"Unsupported ASR engine: {c.engine}")

    c.hf_token = env.get("HF_TOKEN", "")
    c.model = _str(env, "WHISPER_HUB_ENGINE_MODEL", "base")
    c.model_path = _str(env, "WHISPER_HUB_ENGINE_MODEL_PATH", os.path.join(os.path.expanduser("~"), ".cache", "whisper"))

    cuda = None

    def has_cuda() -> bool:
        nonlocal cuda
        if cuda is None:
            cuda = bool(cuda_available())
        return cuda

    c.device = _str(env, "WHISPER_HUB_ENGINE_DEVICE", "") or ("cuda" if has_cuda() else "cpu")
    # Upstream bases the quantization default on CUDA presence, not on the device.
    c.quantization = _str(env, "WHISPER_HUB_ENGINE_QUANTIZATION", "") or ("float32" if has_cuda() else "int8")
    if c.quantization not in QUANTIZATIONS:
        raise ConfigError(
            "Invalid MODEL_QUANTIZATION. Choose 'float32', 'float16', or 'int8' "
            "(or another CTranslate2 compute type: 'bfloat16', 'int16', 'int8_float32', 'int8_float16', 'int8_bfloat16')."
        )

    c.model_idle_timeout = _int(env, "WHISPER_HUB_ENGINE_MODEL_IDLE_TIMEOUT", 0)
    key, level = lookup(env, "WHISPER_HUB_LOG_LEVEL")
    level = (level or "INFO").strip().upper()
    c.log_level = _LEVEL_SPELLINGS.get(level, level)
    if c.log_level not in LOG_LEVELS:
        raise ConfigError(f"{key}={c.log_level!r}: choose one of {', '.join(LOG_LEVELS)}")

    c.subtitle_max_line_width = _int(env, "WHISPER_HUB_ENGINE_SUBTITLE_MAX_LINE_WIDTH", 1000)
    c.subtitle_max_line_count = _int(env, "WHISPER_HUB_ENGINE_SUBTITLE_MAX_LINE_COUNT", 2)
    # Upstream accepts only the literal "true" (any case); everything else is false.
    c.subtitle_highlight_words = _str(env, "WHISPER_HUB_ENGINE_SUBTITLE_HIGHLIGHT_WORDS", "false").lower() == "true"

    c.host = _str(env, "WHISPER_HUB_ENGINE_HOST", "0.0.0.0")
    c.port = _int(env, "WHISPER_HUB_ENGINE_PORT", 9100, minimum=1)
    models = []
    for m in _str(env, "WHISPER_HUB_ENGINE_MODELS", "").split(","):
        m = m.strip()
        if m and m not in models:
            models.append(m)
    c.models = models or [c.model]
    c.max_loaded_models = _int(env, "WHISPER_HUB_ENGINE_MAX_LOADED_MODELS", 1, minimum=1)
    c.preload = _bool(env, "WHISPER_HUB_ENGINE_PRELOAD", False)

    c.fake_segments = _int(env, "WHISPER_HUB_ENGINE_FAKE_SEGMENTS", 3, minimum=0)
    c.fake_delay = _float(env, "WHISPER_HUB_ENGINE_FAKE_DELAY", 0.0)
    c.fake_language = _str(env, "WHISPER_HUB_ENGINE_FAKE_LANGUAGE", "en")
    return c
