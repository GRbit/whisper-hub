"""Worker HTTP API as a plain ASGI application (see docs/worker-api.md).

Plain ASGI instead of a framework because the transcribe stream needs to
watch for http.disconnect itself while it writes, which frameworks hide.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Optional, Tuple

from python_multipart.multipart import MultipartParser, parse_options_header

from . import __version__
from .audio import check_pcm, duration
from .config import Config
from .engines.base import Engine, JobContext, parse_options
from .errors import BadRequest, Internal, WorkerError
from .languages import LANGUAGES
from .models import Job, ModelManager

logger = logging.getLogger("whisper_worker")


def _json_line(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()


async def _send_json(send, status: int, obj) -> None:
    body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def _send_error(send, err: WorkerError) -> None:
    await _send_json(send, err.status, {"error": err.body()})


class _Disconnected(Exception):
    pass


async def _read_multipart(scope, receive) -> Dict[str, bytes]:
    """_read_multipart returns part name -> content. Parts are kept in memory: audio
    becomes a numpy array anyway, and file vs field distinction does not matter here."""
    headers = {k.decode("latin-1").lower(): v for k, v in scope["headers"]}
    ctype, params = parse_options_header(headers.get("content-type", b""))
    if ctype != b"multipart/form-data" or b"boundary" not in params:
        raise BadRequest("expected multipart/form-data")

    parts: Dict[str, bytearray] = {}
    state = {"field": b"", "value": b"", "headers": {}, "name": None}

    def on_part_begin():
        state["headers"] = {}
        state["name"] = None

    def on_header_field(data, start, end):
        state["field"] += data[start:end]

    def on_header_value(data, start, end):
        state["value"] += data[start:end]

    def on_header_end():
        state["headers"][state["field"].lower()] = state["value"]
        state["field"] = b""
        state["value"] = b""

    def on_headers_finished():
        _, disp = parse_options_header(state["headers"].get(b"content-disposition", b""))
        name = disp.get(b"name", b"").decode("utf-8", "replace")
        state["name"] = name
        parts.setdefault(name, bytearray())

    def on_part_data(data, start, end):
        parts[state["name"]] += data[start:end]

    parser = MultipartParser(
        params[b"boundary"],
        {
            "on_part_begin": on_part_begin,
            "on_header_field": on_header_field,
            "on_header_value": on_header_value,
            "on_header_end": on_header_end,
            "on_headers_finished": on_headers_finished,
            "on_part_data": on_part_data,
        },
    )
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            raise _Disconnected()
        try:
            parser.write(message.get("body", b""))
        except Exception:
            raise BadRequest("malformed multipart body") from None
        if not message.get("more_body", False):
            break
    try:
        parser.finalize()
    except Exception:
        raise BadRequest("malformed multipart body") from None
    return {k: bytes(v) for k, v in parts.items()}


def _parse_request(parts: Dict[str, bytes]) -> Tuple[dict, bytes]:
    if "audio" not in parts:
        raise BadRequest("missing audio part")
    raw = parts.get("options", b"")
    try:
        options = json.loads(raw) if raw.strip() else {}
    except ValueError:
        raise BadRequest("options part is not valid JSON") from None
    pcm = parts["audio"]
    check_pcm(pcm)
    return options, pcm


class WorkerApp:
    def __init__(self, cfg: Config, engine: Engine):
        self.cfg = cfg
        self.engine = engine
        self.manager = ModelManager(cfg, engine)
        self._version: Optional[str] = None
        self._jobs = 0

    def version(self) -> str:
        if self._version is None:
            try:
                lib = self.engine.version()
            except Exception:
                logger.debug("engine version lookup failed", exc_info=True)
                lib = "unknown"
            self._version = f"whisper-worker {__version__}; {lib}"
        return self._version

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope["type"] != "http":
            return
        method, path = scope["method"], scope["path"]
        logger.debug("%s %s from %s", method, path, (scope.get("client") or ("?",))[0])
        try:
            if path == "/healthz" and method == "GET":
                await _send_json(send, 200, {"status": "ok"})
            elif path == "/v1/info" and method == "GET":
                await _send_json(send, 200, self.info())
            elif path == "/v1/transcribe" and method == "POST":
                await self.transcribe(scope, receive, send)
            elif path == "/v1/detect-language" and method == "POST":
                await self.detect_language(scope, receive, send)
            elif path in ("/healthz", "/v1/info", "/v1/transcribe", "/v1/detect-language"):
                await _send_json(send, 405, {"error": {"code": "method_not_allowed", "message": "method not allowed"}})
            else:
                await _send_json(send, 404, {"error": {"code": "not_found", "message": "not found"}})
        except _Disconnected:
            logger.debug("%s %s: client disconnected", method, path)

    async def _lifespan(self, receive, send):
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                try:
                    await self.manager.start()
                except Exception:
                    logger.exception("Worker startup failed")
                    await send({"type": "lifespan.startup.failed", "message": "worker startup failed"})
                    return
                logger.info(
                    "Worker ready: engine=%s device=%s models=%s default=%s",
                    self.engine.name,
                    self.cfg.device,
                    ",".join(self.cfg.models),
                    self.cfg.default_model,
                )
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await self.manager.stop()
                await send({"type": "lifespan.shutdown.complete"})
                return

    def info(self) -> dict:
        return {
            "engine": self.engine.name,
            "version": self.version(),
            "device": self.cfg.device,
            "models": self.manager.describe(),
            "languages": list(LANGUAGES),
            "capabilities": dict(self.engine.capabilities),
        }

    async def transcribe(self, scope, receive, send):
        self._jobs += 1
        label = f"job {self._jobs}"
        started = time.monotonic()
        try:
            raw_options, pcm = _parse_request(await _read_multipart(scope, receive))
            opts = parse_options(raw_options)
            model_id = self.manager.resolve(opts.model)
            self.engine.validate(opts)
        except WorkerError as e:
            logger.debug("%s rejected: %s %s", label, e.code, e.message)
            await _send_error(send, e)
            return
        logger.debug(
            "%s: model=%s task=%s language=%s prompt=%s temperature=%s vad=%s words=%s diarize=%s "
            "speakers=%s..%s legacy_output=%s audio=%.2fs",
            label, model_id, opts.task, opts.language, bool(opts.prompt), opts.temperature, opts.vad_filter,
            opts.word_timestamps, opts.diarize, opts.min_speakers, opts.max_speakers, opts.legacy_output,
            duration(pcm),
        )

        loop = asyncio.get_running_loop()
        engine = self.engine
        manager = self.manager

        def work(handle, job: Job):
            ctx = JobContext(manager.emitter(job, loop), job.cancel_event, model_id, engine.name)
            audio = engine.prepare_audio(pcm)
            return engine.transcribe(handle, audio, opts, ctx)

        job = manager.submit(model_id, work, label)
        await self._stream(job, label, started, receive, send, duration(pcm))

    async def _stream(self, job: Job, label: str, started: float, receive, send, audio_seconds: float):
        # The request body is fully read, so the next receive() only returns on disconnect.
        disconnect = asyncio.ensure_future(receive())
        response_started = False
        segments = 0
        finished = False
        try:
            while True:
                getter = asyncio.ensure_future(job.queue.get())
                done, _ = await asyncio.wait({getter, disconnect}, return_when=asyncio.FIRST_COMPLETED)
                if getter not in done:
                    getter.cancel()
                    logger.info("%s: client disconnected after %d segments, stopping", label, segments)
                    return
                kind, value = getter.result()
                if kind == "event":
                    if not response_started:
                        await send(
                            {
                                "type": "http.response.start",
                                "status": 200,
                                "headers": [(b"content-type", b"application/x-ndjson"), (b"cache-control", b"no-cache")],
                            }
                        )
                        response_started = True
                    if value["type"] == "segment":
                        segments += 1
                        logger.debug("%s: segment %d %.2f-%.2f", label, segments, value["segment"]["start"] or 0,
                                     value["segment"]["end"] or 0)
                    elif value["type"] == "info":
                        logger.debug("%s: info language=%s", label, value["language"])
                    await send({"type": "http.response.body", "body": _json_line(value), "more_body": True})
                    continue

                finished = True
                if kind == "result":
                    final = {"type": "done", "text": value.text, "language": value.language}
                    if value.rendered is not None:
                        final["rendered"] = value.rendered
                    if not response_started:
                        # Every engine emits info first; this guards against one that does not.
                        raise Internal("engine produced no info event")
                    await send({"type": "http.response.body", "body": _json_line(final), "more_body": False})
                    logger.info(
                        "%s done: model=%s language=%s audio=%.1fs segments=%d in %.2fs",
                        label, job.model_id, value.language, audio_seconds, segments, time.monotonic() - started,
                    )
                    return
                err: WorkerError = value
                logger.info("%s failed: %s (%s)", label, err.code, err.message)
                if response_started:
                    line = _json_line({"type": "error", "code": err.code, "message": err.message})
                    await send({"type": "http.response.body", "body": line, "more_body": False})
                else:
                    await _send_error(send, err)
                return
        except WorkerError as err:
            if response_started:
                line = _json_line({"type": "error", "code": err.code, "message": err.message})
                await send({"type": "http.response.body", "body": line, "more_body": False})
            else:
                await _send_error(send, err)
        except OSError:
            logger.info("%s: client connection lost", label)
        finally:
            if not finished:
                job.cancel()
            disconnect.cancel()

    async def detect_language(self, scope, receive, send):
        self._jobs += 1
        label = f"detect {self._jobs}"
        started = time.monotonic()
        try:
            raw_options, pcm = _parse_request(await _read_multipart(scope, receive))
            if not isinstance(raw_options, dict):
                raise BadRequest("options must be a JSON object")
            model = raw_options.get("model")
            if model is not None and not isinstance(model, str):
                raise BadRequest("options.model must be a string or null")
            model_id = self.manager.resolve(model)
        except WorkerError as e:
            await _send_error(send, e)
            return
        logger.debug("%s: model=%s audio=%.2fs", label, model_id, duration(pcm))
        engine = self.engine

        def work(handle, job: Job):
            return engine.detect_language(handle, engine.prepare_audio(pcm))

        job = self.manager.submit(model_id, work, label)
        disconnect = asyncio.ensure_future(receive())
        getter = asyncio.ensure_future(job.queue.get())
        try:
            done, _ = await asyncio.wait({getter, disconnect}, return_when=asyncio.FIRST_COMPLETED)
            if getter not in done:
                getter.cancel()
                job.cancel()
                logger.info("%s: client disconnected", label)
                return
            kind, value = getter.result()
            if kind == "error":
                await _send_error(send, value)
                return
            language, confidence = value
            logger.info("%s done: model=%s language=%s in %.2fs", label, model_id, language, time.monotonic() - started)
            await _send_json(
                send,
                200,
                {"language": language, "confidence": float(confidence), "model": model_id, "engine": engine.name},
            )
        finally:
            disconnect.cancel()
