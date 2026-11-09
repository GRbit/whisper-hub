"""Entry point: python -m whisper_worker."""

import logging
import os
import sys

from . import __version__
from .config import ConfigError, from_env
from .engines import create, cuda_probe


def main() -> int:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log = logging.getLogger("whisper_worker")
    try:
        cfg = from_env(cuda_available=cuda_probe(os.environ.get("ASR_ENGINE", "openai_whisper")))
    except ConfigError as e:
        log.error("Invalid configuration: %s", e)
        return 2
    log.setLevel(cfg.log_level)
    # Libraries (httpcore, numba, urllib3) stay at INFO: their DEBUG output buries ours.
    if logging.getLevelName(cfg.log_level) > logging.INFO:
        logging.getLogger().setLevel(cfg.log_level)
    if cfg.engine == "whisperx" and cfg.hf_token == "":
        log.warning("HF_TOKEN is not set: WhisperX diarization is disabled.")
    log.info(
        "whisper-worker %s starting: engine=%s device=%s quantization=%s models=%s listen=%s:%d",
        __version__, cfg.engine, cfg.device, cfg.quantization, ",".join(cfg.models), cfg.host, cfg.port,
    )
    log.debug("model_path=%s idle_timeout=%ds max_loaded=%d preload=%s", cfg.model_path, cfg.model_idle_timeout,
              cfg.max_loaded_models, cfg.preload)

    import uvicorn

    from .app import WorkerApp

    app = WorkerApp(cfg, create(cfg))
    # Access log off: /healthz polling would drown the one-line-per-job info log.
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level=cfg.log_level.lower(), access_log=False,
                lifespan="on", timeout_graceful_shutdown=5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
