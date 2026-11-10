"""Hugging Face URL support for ASR_MODEL / WORKER_MODELS.

A whisper-hub extension: upstream whisper-asr-webservice takes model names
and local paths only. MIT License, Copyright (c) 2022 Ahmet Oner & Besim
Alibegovic.
"""

import logging
import os
import posixpath
from typing import Optional, Tuple
from urllib.parse import urlsplit

try:
    from huggingface_hub import list_repo_files, snapshot_download
except ImportError:  # the fake engine image ships without huggingface_hub

    def list_repo_files(*args, **kwargs):
        raise RuntimeError("huggingface_hub is not installed")

    def snapshot_download(*args, **kwargs):
        raise RuntimeError("huggingface_hub is not installed")


logger = logging.getLogger("whisper_worker")

HF_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co"}

# Same files faster-whisper fetches for a repo id; without the filter a mixed repo
# would also pull its transformers weights (1.6 GB for large-v3-turbo).
CT2_ROOT_PATTERNS = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]


def is_hf_reference(name: str) -> bool:
    """is_hf_reference reports whether name is an http(s) URL on a Hugging Face host."""
    url = urlsplit(name)
    return url.scheme.lower() in ("http", "https") and (url.hostname or "") in HF_HOSTS


def _parse(name: str) -> Tuple[str, Optional[str], str]:
    parts = [p for p in urlsplit(name).path.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Cannot parse Hugging Face URL {name!r}: expected https://huggingface.co/<org>/<repo>")
    repo_id = "/".join(parts[:2])
    rest = parts[2:]
    revision = None
    if rest and rest[0] in ("tree", "blob", "resolve"):
        if len(rest) < 2:
            raise ValueError(f"Hugging Face URL {name!r} has '{rest[0]}' but no revision")
        revision, rest = rest[1], rest[2:]
    return repo_id, revision, "/".join(rest)


def _pick_subdir(repo_id: str, files: list, quantization: str) -> str:
    dirs = sorted({posixpath.dirname(f) for f in files if posixpath.basename(f) == "model.bin"})
    logger.debug("CTranslate2 model dirs in %s: %s", repo_id, dirs)
    if not dirs:
        raise ValueError(
            f"No CTranslate2 weights (model.bin) found in {repo_id}. "
            "Convert the model with ct2-transformers-converter and point ASR_MODEL at the result."
        )
    if "" in dirs:
        return ""
    if len(dirs) == 1:
        return dirs[0]
    # Repos name their variants inconsistently (ct2-int16, ct2_int8_float16), so match loosely.
    wanted = quantization.replace("-", "_")
    matches = [d for d in dirs if wanted in d.replace("-", "_")]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"{repo_id} has several CTranslate2 variants {dirs} and ASR_QUANTIZATION={quantization!r} "
        f"does not select exactly one. Put the folder in the URL, e.g. https://huggingface.co/{repo_id}/tree/main/{dirs[0]}"
    )


def resolve_model(name: str, cache_dir: str, quantization: str, token: Optional[str] = None) -> str:
    """resolve_model turns ASR_MODEL into something faster-whisper can load.

    Hugging Face URLs are downloaded into cache_dir and returned as a local path; values
    without a scheme (model names, repo ids, local paths) are returned unchanged.
    """
    if not is_hf_reference(name):
        # Any other URL would only fail later inside faster-whisper with a confusing error.
        if "://" in name:
            raise ValueError(f"ASR_MODEL={name!r}: only huggingface.co URLs are supported")
        return name

    repo_id, revision, subdir = _parse(name)
    logger.debug("Resolving %s: repo=%s revision=%s subdir=%r", name, repo_id, revision, subdir)
    if not subdir:
        subdir = _pick_subdir(repo_id, list_repo_files(repo_id, revision=revision, token=token), quantization)

    patterns = [f"{subdir}/*"] if subdir else CT2_ROOT_PATTERNS
    logger.info("Downloading %s (revision=%s, folder=%r) into %s", repo_id, revision or "main", subdir, cache_dir)
    root = snapshot_download(repo_id, revision=revision, allow_patterns=patterns, cache_dir=cache_dir, token=token)
    path = os.path.join(root, subdir) if subdir else root
    logger.info("ASR_MODEL %s resolved to %s", name, path)
    return path
