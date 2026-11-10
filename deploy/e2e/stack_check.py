"""Protocol pass through a running whisper-hub: legacy (the user's real
client call shapes), the openai SDK, the wyoming package and a Home
Assistant STT replay. Run by deploy/e2e/stack.sh inside the e2e image.

--mode fake: the Python fake worker (deterministic text, prompt directives
for delays), adds streaming timing, cancellation and priority checks.
--mode real: a real model on /audio/english-front-center.*, checks words.
"""

import argparse
import asyncio
import base64
import http.client
import json
import re
import socket
import sys
import threading
import time
import urllib.parse
import uuid

import openai
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info

sys.path.insert(0, "/interop")
from client import ha_process_audio_stream  # noqa: E402  the wyoming owner's HA replay

FAKE_TEXT = "Hello from the fake engine."
failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f": {detail}" if detail and not cond else ""), flush=True)
    if not cond:
        failures.append(name)


class Target:
    def __init__(self, a):
        u = urllib.parse.urlsplit(a.http)
        self.host, self.port = u.hostname, u.port or 80
        self.base = a.http.rstrip("/")
        self.headers = {}
        if a.basic:
            self.headers["Authorization"] = "Basic " + base64.b64encode(a.basic.encode()).decode()
        self.key = a.key or "unused"
        wh, wp = a.wyoming.split(":")
        self.wyoming = (wh, int(wp))


def multipart(field, filename, data):
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{b}--\r\n".encode()
    return f"multipart/form-data; boundary={b}", body


def post(t, path, filename, data, headers=None, timeout=600):
    ctype, body = multipart("audio_file", filename, data)
    c = http.client.HTTPConnection(t.host, t.port, timeout=timeout)
    c.request("POST", path, body, {"Content-Type": ctype, **t.headers, **(headers or {})})
    r = c.getresponse()
    out = r.status, dict(r.getheaders()), r.read()
    c.close()
    return out


def whisperd_text(body):
    # As whisperd's WhisperClient.kt: "text", else the joined segment texts
    # (whisperx's JSON output, as upstream's, has no top-level text).
    d = json.loads(body)
    return d.get("text", "").strip() or "".join(seg.get("text", "") for seg in d.get("segments", []))


def legacy(t, mode, audio, wav_name):
    want = FAKE_TEXT if mode == "fake" else "front center"
    norm = lambda s: s.lower().replace(",", "").replace(".", "")  # noqa: E731
    for fmt in ("txt", "vtt", "srt", "tsv", "json"):
        st, h, b = post(t, f"/asr?output={fmt}", wav_name, audio)
        check(f"legacy /asr output={fmt}", st == 200 and norm(want) in norm(b.decode()) and h.get("Asr-Engine"),
              (st, b[:200]))
    # go-asr-telegram-transcriber and whisper-go-ui (faster_whisper engine):
    q = "/asr?" + urllib.parse.urlencode({"task": "transcribe", "output": "txt", "encode": "true"})
    for extra, label in (("", "auto language"), ("&language=en", "language"), ("&language=en&vad_filter=true", "language+vad")):
        st, _, b = post(t, q + extra, wav_name, audio, {"Accept": "text/plain"})
        check(f"telegram/whisper-go-ui shape ({label})", st == 200 and norm(want) in norm(b.decode()), (st, b[:200]))
    # whisperd: output=json, reads "text".
    st, _, b = post(t, "/asr?task=transcribe&output=json&encode=true", "rec.wav", audio, {"Accept": "application/json"})
    text = whisperd_text(b) if st == 200 else ""
    check("whisperd shape (output=json text)", norm(want) in norm(text), (st, b[:200]))
    st, _, b = post(t, "/detect-language", wav_name, audio)
    d = json.loads(b) if st == 200 else {}
    check("legacy /detect-language", d.get("language_code") == "en" and d.get("detected_language") == "english", (st, b[:200]))


def sdk(t, mode, path):
    c = openai.OpenAI(base_url=t.base + "/v1", api_key=t.key, max_retries=0)
    want = FAKE_TEXT if mode == "fake" else "front center"
    norm = lambda s: s.lower().replace(",", "").replace(".", "")  # noqa: E731
    f = lambda: open(path, "rb")  # noqa: E731
    r = c.audio.transcriptions.create(model="whisper-1", file=f())
    check("sdk json", norm(want) in norm(r.text), r.text)
    r = c.audio.transcriptions.create(model="whisper-1", file=f(), response_format="verbose_json",
                                      timestamp_granularities=["word", "segment"])
    check("sdk verbose_json words", r.words and r.segments and r.language == "english", r)
    r = c.audio.transcriptions.create(model="whisper-1", file=f(), response_format="srt")
    # Structure, not an exact time: whisperx aligns the first cue to the
    # first word (e.g. 00:00:00,031), the other engines start at 0.
    check("sdk srt", re.match(r"1\n\d\d:\d\d:\d\d,\d{3} --> \d\d:\d\d:\d\d,\d{3}\n\S", r) is not None, r[:80])
    r = c.audio.translations.create(model="whisper-1", file=f())
    check("sdk translation", bool(r.text.strip()), r.text)
    ids = [m.id for m in c.models.list().data]
    check("sdk models.list", "whisper-1" in ids, ids)
    deltas = [e for e in c.audio.transcriptions.create(model="whisper-1", file=f(), stream=True)]
    check("sdk stream", deltas[-1].type == "transcript.text.done" and norm(want) in norm(deltas[-1].text), deltas[-1])
    if mode != "fake":
        return
    # Streaming: deltas must arrive as the engine produces them, not at the end.
    start, times = time.monotonic(), []
    for e in c.audio.transcriptions.create(model="whisper-1", file=f(), stream=True, prompt="fake:delay=0.5,segments=4"):
        if e.type == "transcript.text.delta":
            times.append(time.monotonic() - start)
    check("sdk stream is incremental", len(times) == 4 and times[0] < 1.2 and times[-1] - times[0] > 1.2,
          [round(x, 2) for x in times])


def cancellation(t, audio, wav_name):
    # A client that goes away must stop its job: the fake worker runs one job
    # at a time, so a 30 s job left running would delay the next request.
    ctype, body = multipart("audio_file", wav_name, audio)
    q = "/asr?" + urllib.parse.urlencode({"output": "txt", "initial_prompt": "fake:delay=1,segments=30"})
    s = socket.create_connection((t.host, t.port))
    auth = "".join(f"{k}: {v}\r\n" for k, v in t.headers.items())
    s.sendall(f"POST {q} HTTP/1.1\r\nHost: x\r\nContent-Type: {ctype}\r\nContent-Length: {len(body)}\r\n{auth}\r\n".encode() + body)
    time.sleep(2)
    s.close()
    start = time.monotonic()
    st, _, _ = post(t, "/asr?output=txt", wav_name, audio)
    el = time.monotonic() - start
    check("legacy client disconnect cancels the job", st == 200 and el < 8, (st, round(el, 1)))

    c = openai.OpenAI(base_url=t.base + "/v1", api_key=t.key, max_retries=0)
    stream = c.audio.transcriptions.create(model="whisper-1", file=open("/audio/english-front-center.wav", "rb"),
                                           stream=True, prompt="fake:delay=1,segments=30")
    for e in stream:
        break
    stream.response.close()
    start = time.monotonic()
    c.audio.transcriptions.create(model="whisper-1", file=open("/audio/english-front-center.wav", "rb"))
    el = time.monotonic() - start
    check("openai stream abort cancels the job", el < 8, round(el, 1))


async def wyoming_round_trip(addr, pcm, label, want):
    async with AsyncTcpClient(*addr) as client:
        await client.write_event(Describe().event())
        while True:
            ev = await asyncio.wait_for(client.read_event(), 30)
            if Info.is_type(ev.type):
                info = Info.from_event(ev)
                break
        check("wyoming describe", len(info.asr) == 1 and info.asr[0].models, info)
        await client.write_event(Transcribe(language="en").event())
        await client.write_event(AudioStart(rate=16000, width=2, channels=1).event())
        for i in range(0, len(pcm), 3200):
            await client.write_event(AudioChunk(rate=16000, width=2, channels=1, audio=pcm[i:i + 3200]).event())
        await client.write_event(AudioStop().event())
        while True:
            ev = await asyncio.wait_for(client.read_event(), 120)
            if Transcript.is_type(ev.type) or ev.type == "error":
                break
        text = Transcript.from_event(ev).text if Transcript.is_type(ev.type) else str(ev)
        check(f"wyoming transcript ({label})", want.lower().replace(".", "") in text.lower().replace(",", "").replace(".", ""), text)
    pieces = [pcm[i:i + 320] for i in range(0, len(pcm), 320)]
    state, text = await ha_process_audio_stream(addr, "en", pieces)
    check(f"HA replay ({label})", state == "success" and want.lower().replace(".", "") in text.lower().replace(",", "").replace(".", ""),
          (state, text))


def priority(t, audio, wav_name, pcm):
    # A long legacy job runs, two more wait; a Wyoming utterance sent last
    # must run right after the running one. stack.sh checks the usage log.
    done = []

    def legacy_job(name, prompt):
        q = "/asr?" + urllib.parse.urlencode({"output": "txt", "initial_prompt": prompt})
        st, _, _ = post(t, q, wav_name, audio)
        done.append((name, st))

    threads = [threading.Thread(target=legacy_job, args=("long", "fake:delay=0.5,segments=6"))]
    threads[0].start()
    time.sleep(0.8)
    for n in ("queued1", "queued2"):
        th = threading.Thread(target=legacy_job, args=(n, "fake:delay=0.2,segments=3"))
        th.start()
        threads.append(th)
        time.sleep(0.3)
    asyncio.run(wyoming_utterance(t.wyoming, pcm))
    done.append(("wyoming", 200))
    for th in threads:
        th.join()
    order = [n for n, _ in done]
    check("priority: wyoming overtakes queued jobs", order == ["long", "wyoming", "queued1", "queued2"], order)


async def wyoming_utterance(addr, pcm):
    async with AsyncTcpClient(*addr) as client:
        await client.write_event(AudioStart(rate=16000, width=2, channels=1).event())
        await client.write_event(AudioChunk(rate=16000, width=2, channels=1, audio=pcm).event())
        await client.write_event(AudioStop().event())
        while True:
            ev = await asyncio.wait_for(client.read_event(), 60)
            if Transcript.is_type(ev.type) or ev.type == "error":
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", required=True)
    ap.add_argument("--wyoming", required=True)
    ap.add_argument("--mode", choices=("fake", "real"), required=True)
    ap.add_argument("--basic", help="user:password for legacy paths")
    ap.add_argument("--key", help="API key for the OpenAI paths")
    ap.add_argument("--only", default="legacy,sdk,wyoming,cancel,priority")
    a = ap.parse_args()
    t = Target(a)
    only = a.only.split(",")
    wav = open("/audio/english-front-center.wav", "rb").read()
    pcm = open("/audio/english-front-center.pcm", "rb").read()
    want = FAKE_TEXT if a.mode == "fake" else "front center"
    if "legacy" in only:
        legacy(t, a.mode, wav, "english-front-center.wav")
        m4a = open("/audio/german-trailing-moov.m4a", "rb").read()
        st, _, b = post(t, "/asr?task=transcribe&output=json&encode=true", "rec.m4a", m4a)
        text = whisperd_text(b) if st == 200 else ""
        check("m4a with trailing moov (MediaMuxer layout)", st == 200 and len(text.split()) >= (3 if a.mode == "real" else 1),
              (st, b[:200]))
        if a.mode == "real":
            st, _, b = post(t, "/detect-language", "german.wav", open("/audio/german-20s.wav", "rb").read())
            check("detect-language german", st == 200 and json.loads(b).get("language_code") == "de", b[:200])
    if "sdk" in only:
        sdk(t, a.mode, "/audio/english-front-center.wav")
    if "wyoming" in only:
        asyncio.run(wyoming_round_trip(t.wyoming, pcm, a.mode, want))
    if a.mode == "fake" and "cancel" in only:
        cancellation(t, wav, "english-front-center.wav")
    if a.mode == "fake" and "priority" in only:
        priority(t, wav, "english-front-center.wav", pcm)
    print(f"{len(failures)} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
