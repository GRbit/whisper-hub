"""Measures how soon a worker frees its model after a transcription is canceled.

Runs a long recording A and a short one B once each uninterrupted, then
several times: start A, close its connection after an offset (what the
gateway does to cancel), and send B at once. B's extra time over its
uninterrupted run is the cancel latency: how long the canceled A still held
the model. B's output must equal its uninterrupted run (the model is usable
after a cancel in the middle of a stage), and a last uninterrupted run of A
must equal the first.

  python3 cancel_check.py --worker http://127.0.0.1:9100 --long a.pcm --short b.pcm [--points 8]

Inputs are raw PCM, s16le mono 16 kHz. Prints a digest of A's output, so
runs against two builds can be compared. Exit status 1 when an output
differs. Standard library only, so it runs in any worker image.
"""

import argparse
import hashlib
import http.client
import json
import socket
import sys
import time
import urllib.parse
import uuid


def multipart(options, audio):
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"options\"\r\nContent-Type: application/json\r\n\r\n"
            + json.dumps(options)
            + f"\r\n--{b}\r\nContent-Disposition: form-data; name=\"audio\"\r\nContent-Type: application/octet-stream\r\n\r\n"
            ).encode() + audio + f"\r\n--{b}--\r\n".encode()
    return f"multipart/form-data; boundary={b}", body


def run(host, port, options, audio):
    ctype, body = multipart(options, audio)
    c = http.client.HTTPConnection(host, port, timeout=7200)
    c.request("POST", "/v1/transcribe", body=body, headers={"Content-Type": ctype})
    r = c.getresponse()
    data = r.read()
    c.close()
    if r.status != 200:
        raise RuntimeError(f"transcribe: {r.status} {data[:200]!r}")
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def start_and_drop(host, port, options, audio, after):
    ctype, body = multipart(options, audio)
    s = socket.create_connection((host, port))
    head = (f"POST /v1/transcribe HTTP/1.1\r\nHost: worker\r\nContent-Type: {ctype}\r\n"
            f"Content-Length: {len(body)}\r\n\r\n").encode()
    s.sendall(head + body)
    time.sleep(after)
    s.close()


def digest(lines):
    return hashlib.sha256(json.dumps(lines, sort_keys=True).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default="http://127.0.0.1:9100")
    ap.add_argument("--long", required=True)
    ap.add_argument("--short", required=True)
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--output", default="json", help="legacy_output of every run")
    ap.add_argument("--points", type=int, default=8, help="cancel offsets, spread over A's run time")
    ap.add_argument("--at", default="", help="extra cancel offsets in seconds, comma separated")
    a = ap.parse_args()
    u = urllib.parse.urlsplit(a.worker)
    host, port = u.hostname, u.port or 80
    long_pcm, short_pcm = open(a.long, "rb").read(), open(a.short, "rb").read()
    opts = {"model": a.model, "legacy_output": a.output}

    t = time.monotonic()
    ref_long = run(host, port, opts, long_pcm)
    long_s = time.monotonic() - t
    shorts = []
    for _ in range(3):
        t = time.monotonic()
        ref_short = run(host, port, opts, short_pcm)
        shorts.append(time.monotonic() - t)
    short_s = min(shorts)
    print(f"A: {len(long_pcm) / 32000:.0f}s of audio in {long_s:.1f}s, digest {digest(ref_long)}; "
          f"B alone {short_s:.2f}s", flush=True)

    ok, worst = True, 0.0
    offsets = [long_s * (i + 0.5) / a.points for i in range(a.points)]
    offsets += [float(x) for x in a.at.split(",") if x.strip()]
    for after in sorted(offsets):
        start_and_drop(host, port, opts, long_pcm, after)
        t = time.monotonic()
        got = run(host, port, opts, short_pcm)
        latency = max(0.0, time.monotonic() - t - short_s)
        worst = max(worst, latency)
        same = got == ref_short
        ok = ok and same
        print(f"{'ok  ' if same else 'FAIL'} cancel at {after:6.1f}s: model free {latency:6.2f}s later; "
              f"B identical={same}", flush=True)
    again = run(host, port, opts, long_pcm)
    same = again == ref_long
    ok = ok and same
    print(f"{'ok  ' if same else 'FAIL'} A after the cancels identical={same}; worst cancel latency {worst:.2f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
