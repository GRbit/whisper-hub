import os

import pytest

from whisper_worker.config import ConfigError, from_env
from whisper_worker.model_resolver import is_hf_reference


def test_upstream_defaults_on_cpu():
    c = from_env({})
    assert c.engine == "openai_whisper"
    assert c.model == "base"
    assert c.models == ["base"] and c.default_model == "base"
    assert c.model_path == os.path.join(os.path.expanduser("~"), ".cache", "whisper")
    assert c.device == "cpu"
    assert c.quantization == "int8"
    assert c.model_idle_timeout == 0
    assert c.log_level == "INFO"
    assert (c.subtitle_max_line_width, c.subtitle_max_line_count, c.subtitle_highlight_words) == (1000, 2, False)
    assert (c.host, c.port, c.max_loaded_models, c.preload) == ("0.0.0.0", 9100, 1, False)


def test_cuda_defaults_follow_availability():
    c = from_env({}, cuda_available=lambda: True)
    assert (c.device, c.quantization) == ("cuda", "float32")
    # Upstream keys the quantization default on CUDA presence, not on ASR_DEVICE.
    c = from_env({"ASR_DEVICE": "cpu"}, cuda_available=lambda: True)
    assert (c.device, c.quantization) == ("cpu", "float32")


def test_cuda_probe_not_called_when_both_set():
    def boom():
        raise AssertionError("probe called")

    c = from_env({"ASR_DEVICE": "cpu", "ASR_QUANTIZATION": "int8"}, cuda_available=boom)
    assert c.device == "cpu"


def test_explicit_values():
    c = from_env(
        {
            "ASR_ENGINE": "faster_whisper",
            "ASR_MODEL": "tiny",
            "ASR_MODEL_PATH": "/models",
            "ASR_QUANTIZATION": "int8_float16",
            "MODEL_IDLE_TIMEOUT": "300",
            "HF_TOKEN": "t",
            "LOG_LEVEL": "debug",
            "SUBTITLE_MAX_LINE_WIDTH": "42",
            "SUBTITLE_MAX_LINE_COUNT": "1",
            "SUBTITLE_HIGHLIGHT_WORDS": "TRUE",
            "WORKER_HOST": "127.0.0.1",
            "WORKER_PORT": "9200",
            "WORKER_MAX_LOADED_MODELS": "2",
            "WORKER_PRELOAD": "yes",
        }
    )
    assert c.engine == "faster_whisper" and c.model_path == "/models" and c.quantization == "int8_float16"
    assert c.model_idle_timeout == 300 and c.hf_token == "t" and c.log_level == "DEBUG"
    assert (c.subtitle_max_line_width, c.subtitle_max_line_count, c.subtitle_highlight_words) == (42, 1, True)
    assert (c.host, c.port, c.max_loaded_models, c.preload) == ("127.0.0.1", 9200, 2, True)


@pytest.mark.parametrize("raw", ["1", "yes", "True "])
def test_highlight_words_accepts_only_literal_true(raw):
    assert from_env({"SUBTITLE_HIGHLIGHT_WORDS": raw}).subtitle_highlight_words is False


def test_models_list_first_is_default_and_dedupes():
    c = from_env({"ASR_MODEL": "base", "WORKER_MODELS": " small, tiny ,small,,"})
    assert c.models == ["small", "tiny"]
    assert c.default_model == "small"


def test_models_accept_hf_urls():
    url = "https://huggingface.co/coriollon/whisper-large-v3-turbo-russian/tree/main/ct2_int8_float16"
    c = from_env({"WORKER_MODELS": f"tiny,{url}"})
    assert c.models == ["tiny", url]
    assert is_hf_reference(c.models[1]) and not is_hf_reference(c.models[0])


def test_invalid_quantization_message_matches_upstream():
    with pytest.raises(ConfigError, match="Invalid MODEL_QUANTIZATION"):
        from_env({"ASR_QUANTIZATION": "int4"})


@pytest.mark.parametrize(
    "env",
    [
        {"ASR_ENGINE": "nemo"},
        {"MODEL_IDLE_TIMEOUT": "soon"},
        {"WORKER_PORT": "0"},
        {"WORKER_MAX_LOADED_MODELS": "0"},
        {"WORKER_PRELOAD": "maybe"},
        {"WORKER_FAKE_DELAY": "-1"},
        {"LOG_LEVEL": "LOUD"},
    ],
)
def test_invalid_values_fail(env):
    with pytest.raises(ConfigError):
        from_env(env)
