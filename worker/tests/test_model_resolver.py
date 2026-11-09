# Ported from whisper-asr-webservice (upstream branch main) tests/test_model_resolver.py,
# MIT License, Copyright (c) 2022 Ahmet Oner & Besim Alibegovic.
import pytest

from whisper_worker import model_resolver
from whisper_worker.model_resolver import resolve_model

CORIOLLON_FILES = [
    "README.md",
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "ct2-int16/config.json",
    "ct2-int16/model.bin",
    "ct2-int16/tokenizer.json",
    "ct2_int8_float16/config.json",
    "ct2_int8_float16/model.bin",
    "ct2_int8_float16/tokenizer.json",
]


class FakeHub:
    """Records hub calls so tests run offline and can assert what would be downloaded."""

    def __init__(self, files):
        self.files = files
        self.listed = []
        self.downloads = []

    def list_repo_files(self, repo_id, revision=None, token=None):
        self.listed.append((repo_id, revision))
        return self.files

    def snapshot_download(self, repo_id, revision=None, allow_patterns=None, cache_dir=None, token=None):
        self.downloads.append({"repo_id": repo_id, "revision": revision, "allow_patterns": allow_patterns})
        return f"/cache/{repo_id}"


@pytest.fixture
def hub(monkeypatch):
    fake = FakeHub(CORIOLLON_FILES)
    monkeypatch.setattr(model_resolver, "list_repo_files", fake.list_repo_files)
    monkeypatch.setattr(model_resolver, "snapshot_download", fake.snapshot_download)
    return fake


def test_url_picks_ct2_subdir_matching_quantization(hub):
    path = resolve_model("https://huggingface.co/coriollon/whisper-large-v3-turbo-russian", "/cache", "int8_float16")

    assert path == "/cache/coriollon/whisper-large-v3-turbo-russian/ct2_int8_float16"
    assert hub.downloads == [{
        "repo_id": "coriollon/whisper-large-v3-turbo-russian",
        "revision": None,
        "allow_patterns": ["ct2_int8_float16/*"],
    }]


def test_url_with_tree_revision_and_subdir_skips_listing(hub):
    path = resolve_model(
        "https://huggingface.co/coriollon/whisper-large-v3-turbo-russian/tree/stage-as-v2/ct2-int16",
        "/cache",
        "int8",
    )

    assert path == "/cache/coriollon/whisper-large-v3-turbo-russian/ct2-int16"
    assert hub.listed == []
    assert hub.downloads[0]["revision"] == "stage-as-v2"
    assert hub.downloads[0]["allow_patterns"] == ["ct2-int16/*"]


@pytest.mark.parametrize("name", [
    "base",
    "Systran/faster-whisper-large-v3",
    "org/repo/ct2_int8_float16",
    "models/ru/ct2",
    "hf.co/org/repo",
    "huggingface.co/org/repo",
])
def test_schemeless_values_pass_through(hub, name):
    assert resolve_model(name, "/cache", "int8") == name
    assert hub.listed == [] and hub.downloads == []


@pytest.mark.parametrize("url", [
    "https://example.com/org/repo/ct2",
    "https://huggingface.co.evil.com/org/repo",
    "ftp://huggingface.co/org/repo",
])
def test_non_hf_urls_are_rejected(hub, url):
    with pytest.raises(ValueError, match="huggingface.co"):
        resolve_model(url, "/cache", "int8")
    assert hub.listed == [] and hub.downloads == []


def test_url_with_ct2_weights_in_root(hub):
    hub.files = ["config.json", "model.bin", "model.safetensors", "tokenizer.json", "vocabulary.json"]

    path = resolve_model("https://huggingface.co/Systran/faster-whisper-small", "/cache", "int8")

    assert path == "/cache/Systran/faster-whisper-small"
    assert "model.bin" in hub.downloads[0]["allow_patterns"]
    assert "model.safetensors" not in hub.downloads[0]["allow_patterns"]


def test_single_ct2_subdir_is_used_regardless_of_quantization(hub):
    hub.files = ["model.safetensors", "ct2/model.bin", "ct2/config.json"]
    assert resolve_model("https://huggingface.co/org/repo", "/cache", "float32") == "/cache/org/repo/ct2"


def test_ambiguous_subdirs_fail_with_choices(hub):
    with pytest.raises(ValueError) as e:
        resolve_model("https://huggingface.co/coriollon/whisper-large-v3-turbo-russian", "/cache", "float32")
    assert "ct2-int16" in str(e.value) and "ct2_int8_float16" in str(e.value)
    assert hub.downloads == []


def test_repo_without_ct2_weights_fails(hub):
    hub.files = ["config.json", "model.safetensors"]
    with pytest.raises(ValueError, match="model.bin"):
        resolve_model("https://huggingface.co/org/transformers-only", "/cache", "int8")
