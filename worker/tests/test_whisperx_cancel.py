"""Cancellation hooks of the whisperx adapter, on stand-ins for the library objects
(the library itself is only in the whisperx image; pause_check.py and
cancel_check.py cover the real thing)."""

import logging
import threading

import pytest

from whisper_worker.engines import whisperx as wx
from whisper_worker.errors import JobCancelled


class Ctx:
    def __init__(self):
        self.event = threading.Event()

    def cancelled(self):
        return self.event.is_set()


class Pipeline:
    """Pipeline stands in for whisperx's FasterWhisperPipeline: transformers calls self._forward."""

    def __init__(self):
        self.calls = 0

    def _forward(self, model_inputs):
        self.calls += 1
        return {"text": model_inputs}

    def run(self, batches):
        return [self._forward(b) for b in batches]


def test_batch_hook_stops_a_canceled_job_between_batches():
    p = Pipeline()
    assert wx.install_batch_hook(p)
    ctx = Ctx()
    with wx.cancel_scope(ctx) as scope:
        assert p.run([1, 2]) == [{"text": 1}, {"text": 2}]
        ctx.event.set()
        with pytest.raises(JobCancelled):
            p.run([3])
    assert p.calls == 2 and scope.count("transcribe batch") == 3


def test_batch_hook_is_inert_outside_a_job_and_installs_once():
    p = Pipeline()
    assert wx.install_batch_hook(p) and wx.install_batch_hook(p)
    assert p.run([1]) == [{"text": 1}] and p.calls == 1


def test_unexpected_pipeline_shape_keeps_stage_cancellation_and_warns(caplog):
    class Other:
        pass

    with caplog.at_level(logging.WARNING, logger="whisper_worker"):
        assert not wx.install_batch_hook(Other())
    assert "stage boundaries" in caplog.text


def test_align_model_proxy_checks_between_segments():
    seen = []

    class Model:
        sample_rate = 16000

        def __call__(self, x, lengths=None):
            seen.append(x)
            return x, lengths

    ctx = Ctx()
    m = wx.CancelableModel(Model())
    assert m.sample_rate == 16000
    with wx.cancel_scope(ctx) as scope:
        assert m("a", lengths=1) == ("a", 1)
        ctx.event.set()
        with pytest.raises(JobCancelled):
            m("b")
    assert seen == ["a"] and scope.count("align segment") == 2


def test_scope_reports_a_hook_that_never_ran(caplog):
    with caplog.at_level(logging.WARNING, logger="whisper_worker"):
        with wx.cancel_scope(Ctx()) as scope:
            pass
        scope.expect("transcribe batch")
    assert "never ran" in caplog.text


class Inference:
    """Inference stands in for pyannote's Inference: slide() calls self.infer per batch of chunks."""

    def __init__(self):
        self.batches = 0

    def infer(self, chunks):
        self.batches += 1
        return chunks

    def slide(self, n):
        return [self.infer(i) for i in range(n)]


class Vad:
    def __init__(self):
        self.vad_pipeline = type("P", (), {})()
        self.vad_pipeline._segmentation = Inference()


def test_vad_hook_stops_between_chunk_batches():
    p = Pipeline()
    p.vad_model = Vad()
    assert wx.install_vad_hook(p)
    seg = p.vad_model.vad_pipeline._segmentation
    ctx = Ctx()
    with wx.cancel_scope(ctx) as scope:
        assert seg.slide(2) == [0, 1]
        ctx.event.set()
        with pytest.raises(JobCancelled):
            seg.slide(1)
    assert seg.batches == 2 and scope.count("vad batch") == 3


def test_vad_of_another_shape_warns(caplog):
    p = Pipeline()
    p.vad_model = object()
    with caplog.at_level(logging.WARNING, logger="whisper_worker"):
        assert not wx.install_vad_hook(p)
    assert "stage boundaries" in caplog.text
