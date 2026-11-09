#!/usr/bin/env python3
"""Diff legacy output of a running worker against a running upstream whisper-asr-webservice.

Both get the same raw PCM (upstream with encode=false), so decoding cannot differ.
Stdlib only, so it runs on the host:

    python3 worker/scripts/fidelity.py --ref http://127.0.0.1:19110 --worker http://127.0.0.1:19111 \
        audio/english.pcm audio/german.pcm

Exit status 1 when any body differs. --out DIR keeps both bodies per case.
"""

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
import uuid

FORMATS = ("txt", "vtt", "srt", "tsv", "json")

# name -> (upstream query params, worker options); every case runs for every format.
CASES = {
    "default": ({}, {}),
    "words": ({"word_timestamps": "true"}, {"word_timestamps": True}),
    "vad": ({"vad_filter": "true"}, {"vad_filter": True}),
    "lang-en": ({"language": "en"}, {"language": "en"}),
    "translate": ({"task": "translate"}, {"task": "translate"}),
    "prompt": ({"initial_prompt": "Wikipedia Artikel"}, {"prompt": "Wikipedia Artikel"}),
}


def multipart(fields):
    boundary = uuid.uuid4().hex
    body = b""
    for name, (filename, ctype, data) in fields.items():
        disp = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
        body += f"--{boundary}\r\nContent-Disposition: {disp}\r\nContent-Type: {ctype}\r\n\r\n".encode() + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return f"multipart/form-data; boundary={boundary}", body


def post(url, fields):
    ctype, body = multipart(fields)
    req = urllib.request.Request(url, data=body, headers={"Content-Type": ctype}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def upstream(base, pcm, params, fmt):
    q = urllib.parse.urlencode({"encode": "false", "output": fmt, **params})
    return post(f"{base}/asr?{q}", {"audio_file": ("a.pcm", "application/octet-stream", pcm)})


def worker(base, pcm, options, fmt):
    opts = json.dumps({**options, "legacy_output": fmt}).encode()
    raw = post(f"{base}/v1/transcribe", {
        "options": (None, "application/json", opts),
        "audio": (None, "application/octet-stream", pcm),
    })
    last = json.loads(raw.decode().splitlines()[-1])
    if last.get("type") != "done":
        raise RuntimeError(f"worker failed: {last}")
    return last["rendered"].encode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--worker", required=True)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--out")
    ap.add_argument("audio", nargs="+")
    a = ap.parse_args()

    failed = 0
    for path in a.audio:
        pcm = open(path, "rb").read()
        clip = os.path.splitext(os.path.basename(path))[0]
        for case in a.cases.split(","):
            params, options = CASES[case]
            for fmt in FORMATS:
                name = f"{clip}.{case}.{fmt}"
                try:
                    want = upstream(a.ref, pcm, params, fmt)
                except Exception as e:  # upstream itself fails for some engine/option pairs
                    print(f"SKIP {name}: upstream error {e}")
                    continue
                got = worker(a.worker, pcm, options, fmt)
                same = want == got
                failed += not same
                print(f"{'OK  ' if same else 'DIFF'} {name} ({len(want)} bytes)")
                if a.out:
                    os.makedirs(a.out, exist_ok=True)
                    open(os.path.join(a.out, name + ".ref"), "wb").write(want)
                    open(os.path.join(a.out, name + ".worker"), "wb").write(got)
    print(f"{failed} differing bodies")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
