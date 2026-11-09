#!/usr/bin/env python3
"""Legacy fidelity through the whole stack: the same uploads, in their
original containers, to the reference whisper-asr-webservice and to
whisper-hub, compared on status, headers (all but Date) and body.

    python3 deploy/e2e/fidelity.py --ref http://127.0.0.1:19010 --hub http://127.0.0.1:19020 a.wav b.m4a

--encode false sends the files as raw PCM (they must be 16 kHz s16le) with
encode=false, which takes ffmpeg out of the comparison. Stdlib only.
Exit status 1 when anything differs.
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


def compare(name, ref, hub, out):
    (rs, rh, rb), (hs, hh, hb) = ref, hub
    problems = []
    if rs != hs:
        problems.append(f"status {rs} != {hs}")
    if headers(rh) != headers(hh):
        problems.append(f"headers {headers(rh)} != {headers(hh)}")
    if rb != hb:
        problems.append(f"body differs ({len(rb)} vs {len(hb)} bytes)")
    print(("OK   " if not problems else "DIFF ") + name + "".join("\n     " + p for p in problems))
    if out and problems:
        os.makedirs(out, exist_ok=True)
        open(os.path.join(out, name + ".ref"), "wb").write(rb)
        open(os.path.join(out, name + ".hub"), "wb").write(hb)
    return not problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--hub", required=True)
    ap.add_argument("--encode", default="true", choices=("true", "false"))
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--out")
    ap.add_argument("audio", nargs="+")
    a = ap.parse_args()
    total = same = 0
    for path in a.audio:
        data = open(path, "rb").read()
        fn = os.path.basename(path)
        for case in a.cases.split(","):
            for fmt in FORMATS:
                q = urllib.parse.urlencode({"encode": a.encode, "output": fmt, **CASES[case]})
                total += 1
                same += compare(f"{fn}.{case}.{fmt}", post(f"{a.ref}/asr?{q}", fn, data),
                                post(f"{a.hub}/asr?{q}", fn, data), a.out)
        q = urllib.parse.urlencode({"encode": a.encode})
        total += 1
        same += compare(f"{fn}.detect-language", post(f"{a.ref}/detect-language?{q}", fn, data),
                        post(f"{a.hub}/detect-language?{q}", fn, data), a.out)
    print(f"{same}/{total} identical")
    return 0 if same == total else 1


if __name__ == "__main__":
    sys.exit(main())
