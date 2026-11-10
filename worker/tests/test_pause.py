"""POST /v1/pause: a job pauses at its next pause point while a pause request is open,
another job may use its model meanwhile, and it continues unchanged afterwards."""

import json
import time


def lines_of(resp):
    while True:
        line = resp.readline()
        if not line:
            return
        yield json.loads(line)


def open_pause(worker, job_id, model="tiny"):
    """open_pause returns (connection, response, first line) of a held pause request."""
    c = worker.conn()
    body = json.dumps({"job_id": job_id, "model": model})
    c.request("POST", "/v1/pause", body=body, headers={"Content-Type": "application/json"})
    r = c.getresponse()
    first = json.loads(r.readline()) if r.status == 200 else json.loads(r.read())
    return c, r, first


def without_pause_events(events):
    return [e for e in events if e["type"] not in ("paused", "resumed")]


def test_pause_lends_the_model_and_resumes_unchanged(worker):
    opts = {"prompt": "fake:delay=0.2,segments=5", "legacy_output": "srt"}
    _, _, reference = worker.post("/v1/transcribe", opts)

    c, resp = worker.open_stream({**opts, "job_id": "a"})
    events = []
    it = lines_of(resp)
    events.append(next(it))  # info
    events.append(next(it))  # first segment
    pc, presp, first = open_pause(worker, "a")
    assert presp.status == 200 and first == {"status": "pausing"}
    ev = next(it)
    while ev["type"] == "segment":  # at most the one already being decoded
        events.append(ev)
        ev = next(it)
    assert ev == {"type": "paused"}
    events.append(ev)

    # The paused job holds no model: a second job on it runs to the end at once.
    start = time.monotonic()
    status, _, other = worker.post("/v1/transcribe", {"prompt": "fake:segments=2"})
    assert status == 200 and other[-1]["type"] == "done" and time.monotonic() - start < 2

    pc.close()
    rest = list(it)
    c.close()
    assert rest[0] == {"type": "resumed"}
    events += rest
    assert without_pause_events(events) == reference


def test_pause_registered_before_the_job_arrives(worker):
    pc, presp, _ = open_pause(worker, "early")
    assert presp.status == 200
    c, resp = worker.open_stream({"job_id": "early", "prompt": "fake:segments=2"})
    it = lines_of(resp)
    assert next(it) == {"type": "paused"}
    pc.close()
    rest = list(it)
    c.close()
    assert rest[0] == {"type": "resumed"} and rest[-1]["type"] == "done"
    assert [e["type"] for e in rest] == ["resumed", "info", "segment", "segment", "done"]


def test_disconnect_while_paused_cancels_and_frees_model(worker):
    c, resp = worker.open_stream({"job_id": "gone", "prompt": "fake:delay=0.1,segments=50"})
    it = lines_of(resp)
    next(it)
    pc, presp, _ = open_pause(worker, "gone")
    for ev in it:
        if ev["type"] == "paused":
            break
    c.close()
    # The pause request ends when its job does, and the model is free again.
    assert presp.read() == b""
    pc.close()
    start = time.monotonic()
    status, _, events = worker.post("/v1/transcribe", {"prompt": "fake:segments=1"})
    assert status == 200 and events[-1]["type"] == "done" and time.monotonic() - start < 2


def test_engine_that_ignores_pause_never_reports_paused(worker):
    c, resp = worker.open_stream({"job_id": "short", "prompt": "fake:delay=0.3,segments=2,pause=ignore"})
    it = lines_of(resp)
    next(it)
    pc, presp, _ = open_pause(worker, "short")
    events = list(it)
    c.close()
    assert "paused" not in [e["type"] for e in events] and events[-1]["type"] == "done"
    assert presp.read() == b""
    pc.close()


def test_pause_refused_when_next_model_would_not_fit(worker_factory):
    w = worker_factory(WHISPER_HUB_ENGINE_MODELS="tiny,base", WHISPER_HUB_ENGINE_MAX_LOADED_MODELS="1")
    c, resp = w.open_stream({"job_id": "t", "model": "tiny", "prompt": "fake:delay=0.2,segments=10"})
    next(lines_of(resp))
    pc, presp, body = open_pause(w, "t", model="base")
    assert presp.status == 409 and body["error"]["code"] == "not_pausable"
    pc.close()
    pc, presp, _ = open_pause(w, "t", model="tiny")
    assert presp.status == 200
    pc.close()
    c.close()


def test_pause_request_validation(worker):
    c = worker.conn()
    c.request("POST", "/v1/pause", body=b'{"job_id": 3}', headers={"Content-Type": "application/json"})
    r = c.getresponse()
    assert r.status == 400 and json.loads(r.read())["error"]["code"] == "bad_request"
    c.close()
    c = worker.conn()
    c.request("GET", "/v1/pause")
    r = c.getresponse()
    assert r.status == 405
    r.read()
    c.close()
