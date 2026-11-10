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
    # Upstream keys the quantization default on CUDA presence, not on the device.
    c = from_env({"WHISPER_HUB_ENGINE_DEVICE": "cpu"}, cuda_available=lambda: True)
    assert (c.device, c.quantization) == ("cpu", "float32")


def test_cuda_probe_not_called_when_both_set():
    def boom():
        raise AssertionError("probe called")

    c = from_env({"WHISPER_HUB_ENGINE_DEVICE": "cpu", "WHISPER_HUB_ENGINE_QUANTIZATION": "int8"}, cuda_available=boom)
    assert c.device == "cpu"


def test_explicit_values():
    c = from_env(
        {
            "WHISPER_HUB_ENGINE": "faster_whisper",
            "WHISPER_HUB_ENGINE_MODEL": "tiny",
            "WHISPER_HUB_ENGINE_MODEL_PATH": "/models",
            "WHISPER_HUB_ENGINE_QUANTIZATION": "int8_float16",
            "WHISPER_HUB_ENGINE_MODEL_IDLE_TIMEOUT": "300",
            "HF_TOKEN": "t",
            "WHISPER_HUB_LOG_LEVEL": "debug",
            "WHISPER_HUB_ENGINE_SUBTITLE_MAX_LINE_WIDTH": "42",
            "WHISPER_HUB_ENGINE_SUBTITLE_MAX_LINE_COUNT": "1",
            "WHISPER_HUB_ENGINE_SUBTITLE_HIGHLIGHT_WORDS": "TRUE",
            "WHISPER_HUB_ENGINE_HOST": "127.0.0.1",
            "WHISPER_HUB_ENGINE_PORT": "9200",
            "WHISPER_HUB_ENGINE_MAX_LOADED_MODELS": "2",
            "WHISPER_HUB_ENGINE_PRELOAD": "yes",
        }
    )
    assert c.engine == "faster_whisper" and c.model_path == "/models" and c.quantization == "int8_float16"
    assert c.model_idle_timeout == 300 and c.hf_token == "t" and c.log_level == "DEBUG"
    assert (c.subtitle_max_line_width, c.subtitle_max_line_count, c.subtitle_highlight_words) == (42, 1, True)
    assert (c.host, c.port, c.max_loaded_models, c.preload) == ("127.0.0.1", 9200, 2, True)


@pytest.mark.parametrize("raw", ["1", "yes", "True "])
def test_highlight_words_accepts_only_literal_true(raw):
    assert from_env({"WHISPER_HUB_ENGINE_SUBTITLE_HIGHLIGHT_WORDS": raw}).subtitle_highlight_words is False


def test_models_list_first_is_default_and_dedupes():
    c = from_env({"WHISPER_HUB_ENGINE_MODEL": "base", "WHISPER_HUB_ENGINE_MODELS": " small, tiny ,small,,"})
    assert c.models == ["small", "tiny"]
    assert c.default_model == "small"


def test_models_accept_hf_urls():
    url = "https://huggingface.co/coriollon/whisper-large-v3-turbo-russian/tree/main/ct2_int8_float16"
    c = from_env({"WHISPER_HUB_ENGINE_MODELS": f"tiny,{url}"})
    assert c.models == ["tiny", url]
    assert is_hf_reference(c.models[1]) and not is_hf_reference(c.models[0])


def test_invalid_quantization_message_matches_upstream():
    with pytest.raises(ConfigError, match="Invalid MODEL_QUANTIZATION"):
        from_env({"WHISPER_HUB_ENGINE_QUANTIZATION": "int4"})


@pytest.mark.parametrize(
    "env",
    [
        {"WHISPER_HUB_ENGINE": "nemo"},
        {"WHISPER_HUB_ENGINE_MODEL_IDLE_TIMEOUT": "soon"},
        {"WHISPER_HUB_ENGINE_PORT": "0"},
        {"WHISPER_HUB_ENGINE_MAX_LOADED_MODELS": "0"},
        {"WHISPER_HUB_ENGINE_PRELOAD": "maybe"},
        {"WHISPER_HUB_ENGINE_FAKE_DELAY": "-1"},
        {"WHISPER_HUB_LOG_LEVEL": "LOUD"},
    ],
)
def test_invalid_values_fail(env):
    with pytest.raises(ConfigError):
        from_env(env)


# A whisper-asr-webservice deployment keeps working: its names are aliases.
UPSTREAM = {
    "ASR_ENGINE": "faster_whisper",
    "ASR_MODEL": "tiny",
    "ASR_MODEL_PATH": "/models",
    "ASR_DEVICE": "cuda",
    "ASR_QUANTIZATION": "int8_float16",
    "MODEL_IDLE_TIMEOUT": "300",
    "LOG_LEVEL": "debug",
    "SUBTITLE_MAX_LINE_WIDTH": "42",
    "SUBTITLE_MAX_LINE_COUNT": "1",
    "SUBTITLE_HIGHLIGHT_WORDS": "true",
}


def test_upstream_names_are_aliases():
    c = from_env(UPSTREAM)
    assert (c.engine, c.model, c.models, c.model_path) == ("faster_whisper", "tiny", ["tiny"], "/models")
    assert (c.device, c.quantization, c.model_idle_timeout, c.log_level) == ("cuda", "int8_float16", 300, "DEBUG")
    assert (c.subtitle_max_line_width, c.subtitle_max_line_count, c.subtitle_highlight_words) == (42, 1, True)


def test_whisper_hub_names_win_over_aliases():
    env = dict(UPSTREAM)
    env.update({
        "WHISPER_HUB_ENGINE": "fake",
        "WHISPER_HUB_ENGINE_MODEL": "base",
        "WHISPER_HUB_ENGINE_DEVICE": "cpu",
        "WHISPER_HUB_ENGINE_QUANTIZATION": "int8",
        "WHISPER_HUB_ENGINE_MODEL_IDLE_TIMEOUT": "5",
        "WHISPER_HUB_LOG_LEVEL": "warn",
        "WHISPER_HUB_ENGINE_SUBTITLE_HIGHLIGHT_WORDS": "false",
    })
    c = from_env(env)
    assert (c.engine, c.model, c.device, c.quantization) == ("fake", "base", "cpu", "int8")
    assert (c.model_idle_timeout, c.log_level, c.subtitle_highlight_words) == (5, "WARNING", False)
    # An empty value counts as unset, so the alias still applies.
    assert from_env({"WHISPER_HUB_ENGINE_MODEL": "", "ASR_MODEL": "small"}).model == "small"


def test_old_worker_names_are_gone():
    assert from_env({"WORKER_PORT": "9200", "WORKER_PRELOAD": "true"}).port == 9100


@pytest.mark.parametrize("raw,want", [("warn", "WARNING"), ("Warning", "WARNING"), ("fatal", "CRITICAL"), ("error", "ERROR")])
def test_log_level_takes_the_gateway_spellings(raw, want):
    assert from_env({"WHISPER_HUB_LOG_LEVEL": raw}).log_level == want
