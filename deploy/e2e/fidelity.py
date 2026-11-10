#!/usr/bin/env python3
"""Legacy fidelity through the whole stack: the same uploads, in their
original containers, to the reference (public upstream whisper-asr-webservice,
see deploy/e2e/reference/upstream.env) and to whisper-hub, compared on
status, headers (all but Date and Server) and body.

The expected whisper-hub response is the reference's plus whisper-hub's own
extensions, applied explicitly (with --durations, see expected()):
  - the X-Audio-Duration-Seconds header on every response that decoded audio
  - "audio_duration_seconds" in the /detect-language JSON body
  - uploads that ffmpeg decodes to different audio from a pipe (upstream)
    than from a file (whisper-hub), given by --pipe-differs, are labelled
    EXTENSION: e.g. MP4 with the index at the end, or MP3 whose encoder delay
    is only read from a seekable file

    python3 deploy/e2e/fidelity.py --ref http://127.0.0.1:19010 --hub http://127.0.0.1:19020 a.wav b.m4a

--encode false sends the files as raw PCM (they must be 16 kHz s16le) with
encode=false, which takes ffmpeg out of the comparison. Stdlib only.

Differences are reported, never hidden, in these classes:
  DIFF      unexplained: exit status 1
  EXTENSION a --pipe-differs file: upstream decodes different audio
  KNOWN     a documented upstream behavior (KNOWN_DIFFERENCES, --engine)
  UNSTABLE  the case also differs between two runs of the reference itself
            (--unstable FILE, written by a ref-vs-ref run with --write-diffs)
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid

FORMATS = ("txt", "vtt", "srt", "tsv", "json")
# The same option cases as worker/scripts/fidelity.py.
CASES = {
    "default": {},
    "words": {"word_timestamps": "true"},
    "vad": {"vad_filter": "true"},
    "lang-en": {"language": "en"},
    "translate": {"task": "translate"},
    "prompt": {"initial_prompt": "Wikipedia Artikel"},
}
# Differ by nature of the server, as in internal/legacy's reference test:
# the clock, and uvicorn naming itself where Go sends no Server header.
IGNORED_HEADERS = {"date", "server"}

# engine -> (case, reason): where upstream fails and whisper-hub answers.
KNOWN_DIFFERENCES = {
    "whisperx": [("prompt", "upstream whisperx answers 500 for any initial_prompt (its whisperx rejects the argument); ours works")],
}


def post(url, filename, data):
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio_file\"; filename=\"{filename}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def headers(h):
    return sorted((k.lower(), v) for k, v in h.items() if k.lower() not in IGNORED_HEADERS)


DURATION_HEADER = "x-audio-duration-seconds"


def expected(ref, duration, detect):
    """The reference response with whisper-hub's duration extensions."""
    rs, rh, rb = ref
    if duration is None or rs != 200:
        return ref
    hs = {k.lower(): v for k, v in rh.items()}
    hs[DURATION_HEADER] = duration
    if detect:
        # repr(float) is how both upstream and whisper-hub print JSON numbers.
        rb = rb[:-1] + b',"audio_duration_seconds":' + repr(float(duration)).encode() + b"}"
        hs["content-length"] = str(len(rb))
    return rs, hs, rb


def compare(name, ref, hub, out, known=None, unstable=False, pipe_differs=False):
    (rs, rh, rb), (hs, hh, hb) = ref, hub
    problems = []
    if rs != hs:
        problems.append(f"status {rs} != {hs}")
    if headers(rh) != headers(hh):
        problems.append(f"headers {headers(rh)} != {headers(hh)}")
    if rb != hb:
        problems.append(f"body differs ({len(rb)} vs {len(hb)} bytes)")
    if not problems:
        label = "OK"
    elif pipe_differs:
        label = "EXTENSION"
        problems.append("ffmpeg decodes this upload to different audio from a pipe (upstream) than from a file (whisper-hub)")
    elif known and rs == 500 and hs == 200:
        label = "KNOWN"
        problems.append(known)
    elif unstable:
        label = "UNSTABLE"
        problems.append("the reference itself is not deterministic on this case")
    else:
        label = "DIFF"
    print(f"{label:<9}" + name + "".join("\n          " + p for p in problems))
    if out and problems:
        os.makedirs(out, exist_ok=True)
        open(os.path.join(out, name + ".ref"), "wb").write(rb)
        open(os.path.join(out, name + ".hub"), "wb").write(hb)
    return label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--hub", required=True)
    ap.add_argument("--encode", default="true", choices=("true", "false"))
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--out")
    ap.add_argument("--engine", default="faster_whisper", help="engine under test, for KNOWN_DIFFERENCES")
    ap.add_argument("--unstable", help="file of case names that differ between two reference runs")
    ap.add_argument("--write-diffs", help="write the names of differing cases here (for a ref-vs-ref run)")
    ap.add_argument("--durations", help="JSON file name -> decoded seconds (\"%%.3f\"); applies the extensions")
    ap.add_argument("--pipe-differs", default="", help="comma-separated file names whose pipe decode differs from a file decode")
    ap.add_argument("audio", nargs="+")
    a = ap.parse_args()
    known = dict(KNOWN_DIFFERENCES.get(a.engine, []))
    unstable = set(open(a.unstable).read().split()) if a.unstable else set()
    durations = json.load(open(a.durations)) if a.durations else {}
    pipe_differs = set(filter(None, a.pipe_differs.split(",")))
    counts = {"OK": 0, "EXTENSION": 0, "KNOWN": 0, "UNSTABLE": 0, "DIFF": 0}
    diffs = []

    def run(name, case, ref, hub, fn, detect=False):
        if a.durations:
            ref = expected(ref, durations.get(fn), detect)
        label = compare(name, ref, hub, a.out, known.get(case), name in unstable, fn in pipe_differs)
        counts[label] += 1
        if label != "OK":
            diffs.append(name)

    for path in a.audio:
        data = open(path, "rb").read()
        fn = os.path.basename(path)
        for case in a.cases.split(","):
            for fmt in FORMATS:
                q = urllib.parse.urlencode({"encode": a.encode, "output": fmt, **CASES[case]})
                run(f"{fn}.{case}.{fmt}", case, post(f"{a.ref}/asr?{q}", fn, data), post(f"{a.hub}/asr?{q}", fn, data), fn)
        q = urllib.parse.urlencode({"encode": a.encode})
        run(f"{fn}.detect-language", "detect-language", post(f"{a.ref}/detect-language?{q}", fn, data),
            post(f"{a.hub}/detect-language?{q}", fn, data), fn, detect=True)
    if a.write_diffs:
        open(a.write_diffs, "w").write("\n".join(diffs) + "\n")
    total = sum(counts.values())
    print(f"{counts['OK']}/{total} identical (with whisper-hub's extensions applied), {counts['EXTENSION']} extension(s), "
          f"{counts['KNOWN']} known upstream difference(s), {counts['UNSTABLE']} unstable in the reference, "
          f"{counts['DIFF']} unexplained")
    return 1 if counts["DIFF"] else 0


if __name__ == "__main__":
    sys.exit(main())
