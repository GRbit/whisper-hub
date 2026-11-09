"""PCM helpers. numpy is imported lazily so the fake engine image does not need it."""

from .errors import BadRequest

SAMPLE_RATE = 16000
# 30 s at 16 kHz: the window Whisper uses for language detection.
N_SAMPLES = 30 * SAMPLE_RATE


def check_pcm(pcm: bytes) -> None:
    """check_pcm rejects payloads upstream load_audio would reject with 400."""
    if len(pcm) % 2:
        raise BadRequest(f"Raw PCM payload has an odd length ({len(pcm)} bytes)")
    if not pcm:
        raise BadRequest("Audio is empty")


def duration(pcm: bytes) -> float:
    return (len(pcm) // 2) / SAMPLE_RATE


def to_float32(pcm: bytes):
    """to_float32 converts s16le PCM exactly as upstream app/utils.py load_audio does."""
    import numpy as np

    return np.frombuffer(pcm, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(audio, length: int = N_SAMPLES):
    """pad_or_trim is the numpy branch of whisper.audio.pad_or_trim, without the torch import."""
    import numpy as np

    if audio.shape[-1] > length:
        audio = audio.take(indices=range(length), axis=-1)
    if audio.shape[-1] < length:
        pad_widths = [(0, 0)] * audio.ndim
        pad_widths[-1] = (0, length - audio.shape[-1])
        audio = np.pad(audio, pad_widths)
    return audio
