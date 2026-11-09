import http.client
import json
import socket
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whisper_worker.app import WorkerApp  # noqa: E402
from whisper_worker.config import from_env  # noqa: E402
from whisper_worker.engines import create  # noqa: E402


def multipart(options, audio: bytes, audio_filename=None):
    """multipart builds the two-part body the gateway sends; returns (content_type, body)."""
    boundary = uuid.uuid4().hex
    opt = options if isinstance(options, (bytes, str)) else json.dumps(options)
    if isinstance(opt, str):
        opt = opt.encode()
    disp = 'form-data; name="audio"' + (f'; filename="{audio_filename}"' if audio_filename else "")
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"options\"\r\n"
        f"Content-Type: application/json\r\n\r\n".encode()
        + opt
        + f"\r\n--{boundary}\r\nContent-Disposition: {disp}\r\n"
        f"Content-Type: application/octet-stream\r\n\r\n".encode()
        + audio
        + f"\r\n--{boundary}--\r\n".encode()
    )
    return f"multipart/form-data; boundary={boundary}", body


# One second of silence at 16 kHz s16le.
PCM_1S = b"\x00\x00" * 16000


class Worker:
    """Worker runs the real ASGI app under uvicorn on a free port in a thread."""

    def __init__(self, env):
        import uvicorn

        base = {"ASR_ENGINE": "fake", "ASR_MODEL": "tiny", "LOG_LEVEL": "DEBUG"}
        base.update(env)
        cfg = from_env(base)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.app = WorkerApp(cfg, create(cfg))
        config = uvicorn.Config(self.app, host="127.0.0.1", port=self.port, lifespan="on", log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("worker did not start")
            time.sleep(0.02)

    def conn(self, timeout=10):
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)

    def get(self, path):
        c = self.conn()
        c.request("GET", path)
        r = c.getresponse()
        body = r.read()
        c.close()
        return r.status, json.loads(body)

    def post(self, path, options, audio=PCM_1S, **kw):
        """post returns (status, content_type, lines or json body)."""
        ctype, body = multipart(options, audio, **kw)
        c = self.conn()
        c.request("POST", path, body=body, headers={"Content-Type": ctype})
        r = c.getresponse()
        data = r.read()
        c.close()
        rtype = r.getheader("content-type")
        if rtype == "application/x-ndjson":
            return r.status, rtype, [json.loads(line) for line in data.decode().splitlines()]
        return r.status, rtype, json.loads(data)

    def open_stream(self, options, audio=PCM_1S):
        ctype, body = multipart(options, audio)
        c = self.conn()
        c.request("POST", "/v1/transcribe", body=body, headers={"Content-Type": ctype})
        return c, c.getresponse()

    def stop(self):
        self.server.should_exit = True
        self.thread.join(10)


@pytest.fixture
def worker_factory():
    started = []

    def make(**env):
        w = Worker(env)
        started.append(w)
        return w

    yield make
    for w in started:
        w.stop()


@pytest.fixture
def worker(worker_factory):
    return worker_factory()
