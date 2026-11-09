"""Interop checks of whisper-hub's Wyoming server with the real wyoming package.

Run by run.sh against three fakeserver instances: plain, streaming and failing.
The fake transcript is "heard N.NN seconds", N being the converted audio length.
"""

import asyncio
import math
import struct
import sys
from importlib.metadata import version

from wyoming.asr import Transcribe, Transcript, TranscriptChunk, TranscriptStart, TranscriptStop
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.error import Error
from wyoming.info import Describe, Info

PLAIN = ("wyo-fake-plain", 10300)
STREAM = ("wyo-fake-stream", 10300)
FAIL = ("wyo-fake-fail", 10300)
TIMEOUT = 10

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f": {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def tone(seconds, rate, width, channels):
    """A 440 Hz sine as raw little-endian PCM (8-bit is unsigned, as in WAV)."""
    out = bytearray()
    for i in range(int(seconds * rate)):
        v = math.sin(2 * math.pi * 440 * i / rate) * 0.5
        for _ in range(channels):
            if width == 1:
                out += struct.pack("<B", int(v * 127) + 128)
            elif width == 2:
                out += struct.pack("<h", int(v * 32767))
            else:
                out += struct.pack("<i", int(v * 2147483647))
    return bytes(out)


def chunks(audio, rate, width, channels, samples_per_chunk=1024):
    step = samples_per_chunk * width * channels
    for i in range(0, len(audio), step):
        yield AudioChunk(rate=rate, width=width, channels=channels, audio=audio[i : i + step], timestamp=None)


async def read_until(client, types):
    events = []
    while True:
        ev = await asyncio.wait_for(client.read_event(), TIMEOUT)
        if ev is None:
            events.append(None)
            return events
        events.append(ev)
        if ev.type in types:
            return events


async def describe(addr):
    async with AsyncTcpClient(*addr) as client:
        await client.write_event(Describe().event())
        ev = (await read_until(client, {"info"}))[-1]
        return Info.from_event(ev)


async def utterance(client, audio, rate, width, channels, transcribe=None):
    if transcribe is not None:
        await client.write_event(transcribe.event())
    await client.write_event(AudioStart(rate=rate, width=width, channels=channels).event())
    for ch in chunks(audio, rate, width, channels):
        await client.write_event(ch.event())
    await client.write_event(AudioStop().event())


async def test_describe():
    info = await describe(PLAIN)
    check("info has one asr program", len(info.asr) == 1, info)
    prog = info.asr[0]
    check("program installed", prog.installed)
    check("program name", prog.name == "whisper-hub", prog.name)
    check("models are ids, aliases not listed", [m.name for m in prog.models] == ["tiny", "base.en"], prog.models)
    check("model attribution", prog.models[0].attribution.name == "SYSTRAN", prog.models[0].attribution)
    check("languages include en and zh-HK", {"en", "zh-HK"} <= set(prog.models[0].languages))
    check("streaming off by default", getattr(prog, "supports_transcript_streaming", False) is False)
    check("requires external vad", getattr(prog, "requires_external_vad", True) is True)
    check("other services empty", not (info.tts or info.handle or info.intent or info.wake or info.mic or info.snd))
    sinfo = await describe(STREAM)
    check("streaming advertised when on", getattr(sinfo.asr[0], "supports_transcript_streaming", None) is True)


async def test_round_trip():
    async with AsyncTcpClient(*PLAIN) as client:
        audio = tone(1.5, 44100, 2, 2)
        await utterance(client, audio, 44100, 2, 2, Transcribe(language="en"))
        events = await read_until(client, {"transcript", "error"})
        ev = events[-1]
        check("non-streaming: only transcript", [e.type for e in events] == ["transcript"], [e and e.type for e in events])
        t = Transcript.from_event(ev)
        check("44.1k stereo converted to 1.50 s", t.text == "heard 1.50 seconds", t.text)
        check("transcript language", t.language == "en", t.language)

        # Second utterance on the same connection, no transcribe, 8-bit mono 8 kHz.
        audio = tone(0.75, 8000, 1, 1)
        await utterance(client, audio, 8000, 1, 1)
        t = Transcript.from_event((await read_until(client, {"transcript", "error"}))[-1])
        check("second utterance on one connection", t.text == "heard 0.75 seconds", t.text)

        await client.write_event(Describe().event())
        ev = (await read_until(client, {"info"}))[-1]
        check("describe after utterances", ev.type == "info")


async def test_streaming():
    async with AsyncTcpClient(*STREAM) as client:
        await utterance(client, tone(1.0, 16000, 2, 1), 16000, 2, 1, Transcribe(name="whisper-1", language="nl"))
        events = await read_until(client, {"transcript-stop", "error"})
        types = [e.type for e in events if e is not None]
        want = ["transcript-start", "transcript-chunk", "transcript-chunk", "transcript-chunk", "transcript", "transcript-stop"]
        check("streaming event order", types == want, types)
        start = TranscriptStart.from_event(events[0])
        check("transcript-start language", start.language == "nl", start.language)
        texts = [TranscriptChunk.from_event(e).text for e in events if e.type == "transcript-chunk"]
        check("chunks are snapshots", texts == ["heard", "heard 1.00", "heard 1.00 seconds"], texts)
        final = Transcript.from_event(events[4])
        check("final transcript equals last chunk", final.text == texts[-1], final.text)
        TranscriptStop.from_event(events[-1])


async def ha_process_audio_stream(addr, language, audio_chunks):
    """Replay of homeassistant/components/wyoming/stt.py async_process_audio_stream (dev, Nov 2026)."""

    async def upload(client):
        for audio_bytes in audio_chunks:
            await client.write_event(AudioChunk(rate=16000, width=2, channels=1, audio=audio_bytes).event())
        await client.write_event(AudioStop().event())

    async def receive(client):
        while True:
            event = await client.read_event()
            if event is None:
                return ("error", "connection lost")
            if Error.is_type(event.type):
                err = Error.from_event(event)
                return ("error", f"{err.text} (code: {err.code})")
            if Transcript.is_type(event.type):
                return ("success", Transcript.from_event(event).text)

    async with AsyncTcpClient(*addr) as client:
        await client.write_event(Transcribe(language=language).event())
        await client.write_event(AudioStart(rate=16000, width=2, channels=1).event())
        upload_task = asyncio.create_task(upload(client))
        receive_task = asyncio.create_task(receive(client))
        tasks = (upload_task, receive_task)
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if receive_task in done:
                return receive_task.result()
            upload_task.result()
            return await asyncio.wait_for(receive_task, TIMEOUT)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


async def test_home_assistant():
    # HA's pipeline feeds 16 kHz mono s16 in small chunks (10 ms here).
    audio = tone(2.0, 16000, 2, 1)
    pieces = [audio[i : i + 320] for i in range(0, len(audio), 320)]
    for addr, label in ((PLAIN, "plain"), (STREAM, "streaming")):
        state, text = await ha_process_audio_stream(addr, "en", pieces)
        check(f"HA replay ({label})", (state, text) == ("success", "heard 2.00 seconds"), (state, text))
    state, text = await ha_process_audio_stream(PLAIN, "pt-BR", pieces)
    check("HA replay with raw pipeline locale", state == "success", (state, text))
    state, text = await ha_process_audio_stream(FAIL, "en", pieces)
    check("HA replay sees error event", state == "error" and "forbidden" in text, (state, text))


async def test_disconnect_then_reconnect():
    client = AsyncTcpClient(*PLAIN)
    await client.connect()
    await client.write_event(AudioStart(rate=16000, width=2, channels=1).event())
    await client.write_event(AudioChunk(rate=16000, width=2, channels=1, audio=tone(0.5, 16000, 2, 1)).event())
    await client.disconnect()
    info = await describe(PLAIN)
    check("server alive after mid-utterance disconnect", len(info.asr) == 1)


async def main():
    print(f"wyoming {version('wyoming')}, python {sys.version.split()[0]}")
    for test in (test_describe, test_round_trip, test_streaming, test_home_assistant, test_disconnect_then_reconnect):
        try:
            await test()
        except Exception as err:  # noqa: BLE001 - report and keep going
            check(test.__name__, False, repr(err))
    print(f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
