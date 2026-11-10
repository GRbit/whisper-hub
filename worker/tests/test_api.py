"""Worker API contract, exercised over real HTTP against the fake engine."""

import json
import time

import pytest

from conftest import PCM_1S, multipart


def lines_of(resp):
    """lines_of yields NDJSON objects as the server flushes them."""
    while True:
        line = resp.readline()
        if not line:
            return
        yield json.loads(line)


def test_healthz(worker):
    assert worker.get("/healthz") == (200, {"status": "ok"})


def test_info(worker_factory):
    w = worker_factory(WHISPER_HUB_ENGINE_MODELS="tiny,base")
    status, info = w.get("/v1/info")
    assert status == 200
    assert info["engine"] == "fake"
    assert info["device"] == "cpu"
    assert info["models"] == [
        {"id": "tiny", "default": True, "loaded": False},
        {"id": "base", "default": False, "loaded": False},
    ]
    assert "en" in info["languages"] and "yue" in info["languages"] and len(info["languages"]) == 100
    assert info["capabilities"] == {"word_timestamps": True, "vad_filter": True, "diarize": False, "translate": True}
    assert "whisper-worker" in info["version"]


def test_transcribe_event_order_and_shape(worker):
    status, ctype, events = worker.post("/v1/transcribe", {"model": None}, b"\x00\x00" * 32000)
    assert status == 200 and ctype == "application/x-ndjson"
    assert [e["type"] for e in events] == ["info", "segment", "segment", "segment", "done"]
    assert events[0] == {
        "type": "info", "model": "tiny", "engine": "fake", "language": "en",
        "language_probability": 0.99, "duration": 2.0,
    }
    assert events[1]["segment"] == {
        "id": 1, "seek": 0, "start": 0.0, "end": 2.0, "text": " Hello from the fake engine.", "tokens": [50364],
        "temperature": 0.0, "avg_logprob": -0.25, "compression_ratio": 1.2, "no_speech_prob": 0.01,
        "words": None, "speaker": None,
    }
    assert events[-1] == {
        "type": "done",
        "text": " Hello from the fake engine. This is segment number two. The quick brown fox jumps over the lazy dog.",
        "language": "en",
    }


def test_language_words_and_rendered(worker):
    opts = {"language": "de", "word_timestamps": True, "legacy_output": "srt", "task": "translate"}
    status, _, events = worker.post("/v1/transcribe", opts)
    assert status == 200
    assert events[0]["language"] == "de" and events[0]["language_probability"] == 1.0
    words = events[1]["segment"]["words"]
    assert words[0] == {"word": " Hello", "start": 0.0, "end": 0.4, "probability": 0.9, "speaker": None}
    assert words[-1]["end"] == 2.0
    done = events[-1]
    assert done["language"] == "de"
    assert done["rendered"].startswith("1\n00:00:00,000 --> 00:00:02,000\nHello from the fake engine.\n\n2\n")


def test_audio_part_with_filename_is_accepted(worker):
    status, _, events = worker.post("/v1/transcribe", {}, PCM_1S, audio_filename="a.pcm")
    assert status == 200 and events[-1]["type"] == "done"


def test_segments_directive_and_env_default(worker_factory):
    w = worker_factory(WHISPER_HUB_ENGINE_FAKE_SEGMENTS=5, WHISPER_HUB_ENGINE_FAKE_LANGUAGE="fr")
    _, _, events = w.post("/v1/transcribe", {})
    assert len([e for e in events if e["type"] == "segment"]) == 5
    assert events[0]["language"] == "fr"
    _, _, events = w.post("/v1/transcribe", {"prompt": "fake:segments=0"})
    assert [e["type"] for e in events] == ["info", "done"]
    assert events[-1]["text"] == ""


@pytest.mark.parametrize(
    "options,audio,status,code",
    [
        ({"model": "large"}, PCM_1S, 404, "model_not_found"),
        ({"task": "summarize"}, PCM_1S, 400, "bad_request"),
        ({"language": "xx"}, PCM_1S, 400, "bad_request"),
        ({"legacy_output": "html"}, PCM_1S, 400, "bad_request"),
        ({"temperature": "hot"}, PCM_1S, 400, "bad_request"),
        ({"word_timestamps": "yes"}, PCM_1S, 400, "bad_request"),
        ({}, b"\x00\x00\x00", 400, "bad_request"),
        ({}, b"", 400, "bad_request"),
        (b"{not json", PCM_1S, 400, "bad_request"),
        ({"prompt": "fake:fail=bad_request"}, PCM_1S, 400, "bad_request"),
        ({"prompt": "fake:fail=model_not_found"}, PCM_1S, 404, "model_not_found"),
        ({"prompt": "fake:fail=unavailable"}, PCM_1S, 503, "unavailable"),
        ({"prompt": "fake:fail=internal"}, PCM_1S, 500, "internal"),
        ({"prompt": "fake:bogus=1"}, PCM_1S, 400, "bad_request"),
    ],
)
def test_errors_before_output_are_http_errors(worker, options, audio, status, code):
    got_status, ctype, body = worker.post("/v1/transcribe", options, audio)
    assert (got_status, ctype) == (status, "application/json")
    assert body["error"]["code"] == code
    assert body["error"]["message"]
    assert "/" not in body["error"]["message"] and "Traceback" not in body["error"]["message"]


def test_missing_audio_part(worker):
    body = (
        b"--b\r\nContent-Disposition: form-data; name=\"options\"\r\n\r\n{}\r\n--b--\r\n"
    )
    c = worker.conn()
    c.request("POST", "/v1/transcribe", body=body, headers={"Content-Type": "multipart/form-data; boundary=b"})
    r = c.getresponse()
    assert r.status == 400 and json.loads(r.read())["error"]["code"] == "bad_request"


def test_not_multipart(worker):
    c = worker.conn()
    c.request("POST", "/v1/transcribe", body=b"{}", headers={"Content-Type": "application/json"})
    r = c.getresponse()
    assert r.status == 400


def test_failure_after_stream_started_is_error_line(worker):
    status, _, events = worker.post("/v1/transcribe", {"prompt": "fake:fail_after=2"})
    assert status == 200
    assert [e["type"] for e in events] == ["info", "segment", "segment", "error"]
    assert events[-1]["code"] == "internal" and events[-1]["message"]


def test_segments_stream_as_produced(worker):
    c, r = worker.open_stream({"prompt": "fake:delay=0.4,segments=3"})
    assert r.status == 200
    arrivals = []
    for event in lines_of(r):
        arrivals.append((event["type"], time.monotonic()))
    c.close()
    seg_times = [t for kind, t in arrivals if kind == "segment"]
    assert len(seg_times) == 3
    # Each segment arrives about one delay after the previous one, not all at the end.
    assert seg_times[1] - seg_times[0] > 0.25 and seg_times[2] - seg_times[1] > 0.25


def test_health_and_info_answer_while_job_runs(worker):
    c, r = worker.open_stream({"prompt": "fake:delay=0.5,segments=6"})
    assert r.status == 200
    next(lines_of(r))  # info: the job is now running
    t0 = time.monotonic()
    assert worker.get("/healthz")[0] == 200
    assert worker.get("/v1/info")[1]["models"][0]["loaded"] is True
    assert time.monotonic() - t0 < 0.3
    c.close()


def test_disconnect_stops_job_and_frees_model(worker):
    # Without cancellation this job would hold the model for 100 s.
    c, r = worker.open_stream({"prompt": "fake:delay=1,segments=100"})
    assert r.status == 200
    events = lines_of(r)
    assert next(events)["type"] == "info"
    c.close()
    t0 = time.monotonic()
    status, _, events = worker.post("/v1/transcribe", {})
    assert status == 200 and events[-1]["type"] == "done"
    assert time.monotonic() - t0 < 3


def test_jobs_for_one_model_run_one_at_a_time(worker):
    c1, r1 = worker.open_stream({"prompt": "fake:delay=0.3,segments=3"})
    assert r1.status == 200
    t0 = time.monotonic()
    status, _, events = worker.post("/v1/transcribe", {})
    waited = time.monotonic() - t0
    assert status == 200 and events[-1]["type"] == "done"
    assert waited > 0.5  # queued behind the first job
    assert [e["type"] for e in lines_of(r1)][-1] == "done"
    c1.close()


def test_detect_language(worker_factory):
    w = worker_factory(WHISPER_HUB_ENGINE_FAKE_LANGUAGE="de")
    status, _, body = w.post("/v1/detect-language", {"model": None})
    assert status == 200
    assert body == {"language": "de", "confidence": 0.99, "model": "tiny", "engine": "fake"}
    status, _, body = w.post("/v1/detect-language", {"model": "nope"})
    assert status == 404 and body["error"]["code"] == "model_not_found"


def test_unknown_route_and_method(worker):
    assert worker.get("/nope")[0] == 404
    assert worker.get("/v1/transcribe")[0] == 405


def test_max_loaded_models_evicts_least_recently_used(worker_factory):
    w = worker_factory(WHISPER_HUB_ENGINE_MODELS="tiny,base,small", WHISPER_HUB_ENGINE_MAX_LOADED_MODELS=2)
    for m in ("tiny", "base", "small"):
        assert w.post("/v1/transcribe", {"model": m})[0] == 200
    loaded = {m["id"]: m["loaded"] for m in w.get("/v1/info")[1]["models"]}
    assert loaded == {"tiny": False, "base": True, "small": True}


def test_idle_models_unload(worker_factory):
    w = worker_factory(WHISPER_HUB_ENGINE_MODEL_IDLE_TIMEOUT=1)
    assert w.post("/v1/transcribe", {})[0] == 200
    assert w.get("/v1/info")[1]["models"][0]["loaded"] is True
    deadline = time.monotonic() + 5
    while w.get("/v1/info")[1]["models"][0]["loaded"]:
        assert time.monotonic() < deadline, "model was not unloaded"
        time.sleep(0.1)
    # Unloaded models load again on demand.
    assert w.post("/v1/transcribe", {})[0] == 200


def test_preload(worker_factory):
    w = worker_factory(WHISPER_HUB_ENGINE_PRELOAD="true")
    assert w.get("/v1/info")[1]["models"][0]["loaded"] is True


def test_multipart_helper_matches_gateway_shape():
    ctype, body = multipart({"model": "tiny"}, b"\x01\x00")
    assert ctype.startswith("multipart/form-data; boundary=")
    assert body.index(b'name="options"') < body.index(b'name="audio"')
