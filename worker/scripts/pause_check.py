"""Checks pausing on a running worker with a real engine (docs/worker-api.md, POST /v1/pause).

For a long recording A and a short one B it runs each once uninterrupted,
then runs A again, pauses it, runs B to the end while A is paused, resumes
A, and compares: A's stream (without the paused and resumed lines) and B's
stream must equal their uninterrupted runs line for line, rendered output
included. The pause is requested before A arrives (pause point at the
start) and one second into A (a pause point inside the engine, for whisperx
usually after its transcribe stage).

  python3 pause_check.py --worker http://127.0.0.1:9100 --long a.pcm --short b.pcm

Inputs are raw PCM, s16le mono 16 kHz. Exit status 1 when anything differs.
Standard library only, so it runs in any worker image.
"""

import argparse
import http.client
import json
import sys
import threading
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


class Worker:
    def __init__(self, url):
        u = urllib.parse.urlsplit(url)
        self.host, self.port = u.hostname, u.port or 80

    def stream(self, options, audio):
        ctype, body = multipart(options, audio)
        c = http.client.HTTPConnection(self.host, self.port, timeout=3600)
        c.request("POST", "/v1/transcribe", body=body, headers={"Content-Type": ctype})
        r = c.getresponse()
        if r.status != 200:
            raise RuntimeError(f"transcribe: {r.status} {r.read()[:200]!r}")
        return c, r

    def run(self, options, audio):
        c, r = self.stream(options, audio)
        lines = [json.loads(line) for line in r.read().decode().splitlines() if line.strip()]
        c.close()
        return lines

    def pause(self, job_id, model):
        c = http.client.HTTPConnection(self.host, self.port, timeout=3600)
        body = json.dumps({"job_id": job_id, "model": model})
        c.request("POST", "/v1/pause", body=body, headers={"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            raise RuntimeError(f"pause: {r.status} {r.read()[:200]!r}")
        r.readline()
        return c


def strip(lines):
    return [e for e in lines if e["type"] not in ("paused", "resumed")]


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"line {i}: {json.dumps(x)[:300]} != {json.dumps(y)[:300]}"
    return f"length {len(a)} != {len(b)}"


def scenario(w, name, model, opts, long_pcm, short_pcm, ref_long, ref_short, early):
    job_id = f"check-{name}-{uuid.uuid4().hex[:6]}"
    holder = {}
    if early:
        holder["c"] = w.pause(job_id, model)
    t0 = time.monotonic()
    if not early:
        # Started before the request: whisperx sends its status line only at the end.
        def later():
            time.sleep(1)
            holder["c"] = w.pause(job_id, model)
            holder["at"] = time.monotonic()
        threading.Thread(target=later, daemon=True).start()
    c, r = w.stream({**opts, "job_id": job_id}, long_pcm)
    lines = []
    while True:
        line = r.readline()
        if not line:
            raise RuntimeError("stream ended before the pause took effect")
        ev = json.loads(line)
        lines.append(ev)
        if ev["type"] == "paused":
            break
        if ev["type"] in ("done", "error"):
            raise RuntimeError(f"job finished before the pause took effect ({time.monotonic() - t0:.1f}s)")
    hold = holder["c"]
    pause_asked = holder.get("at")
    took = time.monotonic() - (pause_asked or t0)
    before = len([e for e in lines if e["type"] == "segment"])
    b_start = time.monotonic()
    other = w.run(opts, short_pcm)
    b_time = time.monotonic() - b_start
    hold.close()
    lines += [json.loads(x) for x in r.read().decode().splitlines() if x.strip()]
    c.close()
    ok_b = other == ref_short
    ok_a = strip(lines) == ref_long
    print(f"{'ok  ' if ok_a and ok_b else 'FAIL'} {name}: paused {took:.2f}s after the request, after {before} "
          f"segments; B ran {b_time:.2f}s meanwhile; A identical={ok_a} B identical={ok_b}", flush=True)
    if not ok_a:
        print("     A " + first_diff(strip(lines), ref_long))
    if not ok_b:
        print("     B " + first_diff(other, ref_short))
    return ok_a and ok_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default="http://127.0.0.1:9100")
    ap.add_argument("--long", required=True)
    ap.add_argument("--short", required=True)
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--output", default="json", help="legacy_output of every run")
    ap.add_argument("--words", action="store_true", help="request word timestamps")
    a = ap.parse_args()
    w = Worker(a.worker)
    long_pcm, short_pcm = open(a.long, "rb").read(), open(a.short, "rb").read()
    opts = {"model": a.model, "legacy_output": a.output, "word_timestamps": a.words}
    t = time.monotonic()
    ref_long = w.run(opts, long_pcm)
    print(f"reference A: {len([e for e in ref_long if e['type'] == 'segment'])} segments in {time.monotonic() - t:.1f}s")
    ref_short = w.run(opts, short_pcm)
    again = w.run(opts, long_pcm)
    if again != ref_long:
        print("     the engine is not deterministic on A without any pause: " + first_diff(again, ref_long))
    ok = True
    for name, early in (("pause at start", True), ("pause inside the engine", False)):
        try:
            ok = scenario(w, name, a.model, opts, long_pcm, short_pcm, ref_long, ref_short, early) and ok
        except RuntimeError as e:
            print(f"FAIL {name}: {e}")
            ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
