"""Legacy /asr output writers used by upstream for the faster_whisper engine.

Copied from whisper-asr-webservice (main, see
deploy/e2e/reference/upstream.env): the writer classes from app/utils.py
and the dispatch from app/asr_models/faster_whisper_engine.py write_result. MIT License,
Copyright (c) 2022 Ahmet Oner & Besim Alibegovic.

format_timestamp is copied from faster_whisper/utils.py (faster-whisper
1.2.1, MIT License, Copyright (c) 2023 SYSTRAN) so the fake engine can render
without the library; a test checks it against the original.

Segments are objects with attributes (faster_whisper.transcribe.Segment, or
the look-alike dataclasses below), not dicts: that is what upstream feeds
these writers.
"""

import json
from dataclasses import asdict, dataclass
from io import StringIO
from typing import List, Optional, TextIO


def format_timestamp(
    seconds: float,
    always_include_hours: bool = False,
    decimal_marker: str = ".",
) -> str:
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"


@dataclass
class Word:
    """Word mirrors faster_whisper.transcribe.Word field for field (order matters for JSON)."""

    start: float
    end: float
    word: str
    probability: float


@dataclass
class Segment:
    """Segment mirrors faster_whisper.transcribe.Segment field for field (order matters for JSON)."""

    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: Optional[List[Word]]
    temperature: Optional[float]


class ResultWriter:
    extension: str

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def write_result(self, result: dict, file: TextIO):
        raise NotImplementedError


class WriteTXT(ResultWriter):
    extension: str = "txt"

    def write_result(self, result: dict, file: TextIO):
        for segment in result["segments"]:
            print(segment.text.strip(), file=file, flush=True)


class WriteVTT(ResultWriter):
    extension: str = "vtt"

    def write_result(self, result: dict, file: TextIO):
        print("WEBVTT\n", file=file)
        for segment in result["segments"]:
            print(
                f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n"
                f"{segment.text.strip().replace('-->', '->')}\n",
                file=file,
                flush=True,
            )


class WriteSRT(ResultWriter):
    extension: str = "srt"

    def write_result(self, result: dict, file: TextIO):
        for i, segment in enumerate(result["segments"], start=1):
            print(
                f"{i}\n"
                f"{format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')} --> "
                f"{format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')}\n"
                f"{segment.text.strip().replace('-->', '->')}\n",
                file=file,
                flush=True,
            )


class WriteTSV(ResultWriter):
    """<start ms>\t<end ms>\t<text> lines; integer milliseconds avoid locale decimal commas."""

    extension: str = "tsv"

    def write_result(self, result: dict, file: TextIO):
        print("start", "end", "text", sep="\t", file=file)
        for segment in result["segments"]:
            print(round(1000 * segment.start), file=file, end="\t")
            print(round(1000 * segment.end), file=file, end="\t")
            print(segment.text.strip().replace("\t", " "), file=file, flush=True)


class WriteJSON(ResultWriter):
    extension: str = "json"

    def write_result(self, result: dict, file: TextIO):
        if "segments" in result:
            result["segments"] = [asdict(segment) for segment in result["segments"]]
        json.dump(result, file)


def render(result: dict, output: Optional[str]) -> str:
    """render returns the /asr body upstream faster_whisper produces for result.

    result is {"language", "segments", "text"} in that key order; WriteJSON
    replaces result["segments"], so pass a dict you no longer need.
    """
    file = StringIO()
    # Upstream passes the ResultWriter class as output_dir; it is never used.
    if output == "srt":
        WriteSRT(ResultWriter).write_result(result, file=file)
    elif output == "vtt":
        WriteVTT(ResultWriter).write_result(result, file=file)
    elif output == "tsv":
        WriteTSV(ResultWriter).write_result(result, file=file)
    elif output == "json":
        WriteJSON(ResultWriter).write_result(result, file=file)
    else:
        WriteTXT(ResultWriter).write_result(result, file=file)
    return file.getvalue()
