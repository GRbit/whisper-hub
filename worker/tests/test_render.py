"""Legacy rendering of the faster_whisper writer path (also used by the fake engine).

Byte-level fidelity against the real upstream service is checked by
worker/scripts/fidelity.py; these tests pin the writer behavior offline.
"""

import dataclasses
import json

import pytest

from whisper_worker import legacy_writers as lw
from whisper_worker.engines.fake import fake_segments


def render(output, segments=None, language="en"):
    segments = fake_segments(3, False) if segments is None else segments
    text = "".join(s.text for s in segments)
    return lw.render({"language": language, "segments": segments, "text": text}, output)


def seg(text, start=0.0, end=1.0, words=None):
    return lw.Segment(1, 0, start, end, text, [1], -0.1, 1.0, 0.0, words, 0.0)


def test_txt():
    assert render("txt") == (
        "Hello from the fake engine.\nThis is segment number two.\nThe quick brown fox jumps over the lazy dog.\n"
    )


def test_unknown_or_missing_output_falls_back_to_txt():
    assert render(None) == render("txt")


def test_vtt():
    assert render("vtt") == (
        "WEBVTT\n\n"
        "00:00.000 --> 00:02.000\nHello from the fake engine.\n\n"
        "00:02.000 --> 00:04.000\nThis is segment number two.\n\n"
        "00:04.000 --> 00:06.000\nThe quick brown fox jumps over the lazy dog.\n\n"
    )


def test_srt():
    assert render("srt") == (
        "1\n00:00:00,000 --> 00:00:02,000\nHello from the fake engine.\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\nThis is segment number two.\n\n"
        "3\n00:00:04,000 --> 00:00:06,000\nThe quick brown fox jumps over the lazy dog.\n\n"
    )


def test_tsv():
    assert render("tsv") == (
        "start\tend\ttext\n"
        "0\t2000\tHello from the fake engine.\n"
        "2000\t4000\tThis is segment number two.\n"
        "4000\t6000\tThe quick brown fox jumps over the lazy dog.\n"
    )


def test_json_key_order_and_values():
    out = render("json", fake_segments(1, False))
    assert out == (
        '{"language": "en", "segments": [{"id": 1, "seek": 0, "start": 0.0, "end": 2.0, '
        '"text": " Hello from the fake engine.", "tokens": [50364], "avg_logprob": -0.25, '
        '"compression_ratio": 1.2, "no_speech_prob": 0.01, "words": null, "temperature": 0.0}], '
        '"text": " Hello from the fake engine."}'
    )


def test_json_words_are_nested_dicts_in_dataclass_order():
    out = json.loads(render("json", fake_segments(1, True)))
    words = out["segments"][0]["words"]
    assert list(words[0]) == ["start", "end", "word", "probability"]
    assert [w["word"] for w in words] == [" Hello", " from", " the", " fake", " engine."]


def test_json_escapes_non_ascii_like_upstream():
    # Upstream faster_whisper uses json.dump defaults: ensure_ascii=True.
    out = render("json", [seg(" Gr\u00fc\u00dfe")], language="de")
    assert '"text": " Gr\\u00fc\\u00dfe"' in out


def test_subtitle_arrow_and_tsv_tab_escaping():
    s = [seg(" a --> b\tc ")]
    assert "a -> b\tc\n" in render("srt", s)
    assert "a -> b\tc\n" in render("vtt", s)
    assert render("tsv", s).endswith("0\t1000\ta --> b c\n")


@pytest.mark.parametrize(
    "seconds,hours,marker,want",
    [(0, False, ".", "00:00.000"), (3661.0005, False, ".", "01:01:01.000"), (59.9996, True, ",", "00:01:00,000"),
     (7.5600000000000005, True, ",", "00:00:07,560")],
)
def test_format_timestamp(seconds, hours, marker, want):
    assert lw.format_timestamp(seconds, always_include_hours=hours, decimal_marker=marker) == want


def test_copies_match_faster_whisper_when_installed():
    fw_utils = pytest.importorskip("faster_whisper.utils")
    fw = pytest.importorskip("faster_whisper.transcribe")
    for s in (0, 0.0004, 0.0005, 1.2345, 59.9995, 3599.9996, 36000.5):
        for h in (False, True):
            for m in (".", ","):
                assert lw.format_timestamp(s, h, m) == fw_utils.format_timestamp(s, h, m)
    names = lambda cls: [f.name for f in dataclasses.fields(cls)]  # noqa: E731
    assert names(lw.Segment) == names(fw.Segment)
    assert names(lw.Word) == names(fw.Word)


def test_segment_numbers_are_strict_json_safe():
    from whisper_worker.engines.base import make_segment, make_word

    s = make_segment(id=0, start=float("nan"), end=float("inf"), text="x", avg_logprob=None,
                     words=[make_word(" x", float("-inf"), 1, float("nan"))])
    assert (s["start"], s["end"], s["avg_logprob"]) == (0.0, 0.0, None)
    assert (s["words"][0]["start"], s["words"][0]["probability"]) == (0.0, 0.0)
    json.dumps(s, allow_nan=False)
