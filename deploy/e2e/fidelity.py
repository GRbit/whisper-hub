#!/usr/bin/env python3
"""Legacy fidelity through the whole stack: the same uploads, in their
original containers, to the reference whisper-asr-webservice and to
whisper-hub, compared on status, headers (all but Date) and body.

    python3 deploy/e2e/fidelity.py --ref http://127.0.0.1:19010 --hub http://127.0.0.1:19020 a.wav b.m4a

--encode false sends the files as raw PCM (they must be 16 kHz s16le) with
encode=false, which takes ffmpeg out of the comparison. Stdlib only.

Differences are reported, never hidden, in three classes:
  DIFF      unexplained: exit status 1
  KNOWN     a documented upstream behavior (KNOWN_DIFFERENCES, --engine)
  UNSTABLE  the case also differs between two runs of the reference itself
            (--unstable FILE, written by a ref-vs-ref run with --write-diffs)
"""
import argparse
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid

FORMATS = ("txt", "vtt", "srt", "tsv", "json")
# The worker owner's option cases (worker/scripts/fidelity.py).
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


def compare(name, ref, hub, out, known=None, unstable=False):
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
    ap.add_argument("audio", nargs="+")
    a = ap.parse_args()
    known = dict(KNOWN_DIFFERENCES.get(a.engine, []))
    unstable = set(open(a.unstable).read().split()) if a.unstable else set()
    counts = {"OK": 0, "KNOWN": 0, "UNSTABLE": 0, "DIFF": 0}
    diffs = []

    def run(name, case, ref, hub):
        label = compare(name, ref, hub, a.out, known.get(case), name in unstable)
        counts[label] += 1
        if label != "OK":
            diffs.append(name)

    for path in a.audio:
        data = open(path, "rb").read()
        fn = os.path.basename(path)
        for case in a.cases.split(","):
            for fmt in FORMATS:
                q = urllib.parse.urlencode({"encode": a.encode, "output": fmt, **CASES[case]})
                run(f"{fn}.{case}.{fmt}", case, post(f"{a.ref}/asr?{q}", fn, data), post(f"{a.hub}/asr?{q}", fn, data))
        q = urllib.parse.urlencode({"encode": a.encode})
        run(f"{fn}.detect-language", "detect-language", post(f"{a.ref}/detect-language?{q}", fn, data),
            post(f"{a.hub}/detect-language?{q}", fn, data))
    if a.write_diffs:
        open(a.write_diffs, "w").write("\n".join(diffs) + "\n")
    total = sum(counts.values())
    print(f"{counts['OK']}/{total} identical, {counts['KNOWN']} known upstream difference(s), "
          f"{counts['UNSTABLE']} unstable in the reference, {counts['DIFF']} unexplained")
    return 1 if counts["DIFF"] else 0


if __name__ == "__main__":
    sys.exit(main())
