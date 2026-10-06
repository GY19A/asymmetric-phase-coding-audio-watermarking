"""Audio file I/O for the tools. The library itself works on arrays only."""
from __future__ import annotations

import os
import wave
from math import gcd

import numpy as np

from .params import SR


class AudioIOError(OSError):
    pass


def read_audio(path, notices: list | None = None):
    """(mono float64 samples, sample rate). PCM16 reads as int16 / 32768 exactly. Uses
    soundfile when installed (any libsndfile format), else the stdlib wave module (PCM16 WAV).
    Multichannel input is downmixed by the channel mean; the fact is appended to `notices`."""
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise AudioIOError(f"cannot read {path}: no such file")
    try:
        import soundfile as sf
    except ImportError:
        sf = None
    try:
        if sf is not None:
            data, sr = sf.read(path, dtype="float64", always_2d=True)
        else:
            with wave.open(path, "rb") as w:
                if w.getsampwidth() != 2:
                    raise AudioIOError(f"{path}: only 16-bit PCM WAV is supported without soundfile")
                sr, ch = w.getframerate(), w.getnchannels()
                raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
            data = raw.reshape(-1, ch).astype(np.float64) / 32768.0
    except AudioIOError:
        raise
    except Exception as e:                              # libsndfile / wave format errors
        raise AudioIOError(f"cannot read {path}: {e}") from e
    if data.shape[1] > 1:
        if notices is not None:
            notices.append(f"downmixed {data.shape[1]} channels to mono (channel mean)")
        x = data.mean(axis=1)
    else:
        x = data[:, 0].copy()
    return x, int(sr)


def to_44100(x: np.ndarray, sr: int, notices: list | None = None) -> np.ndarray:
    """Format §1: other rates are resampled to 44100 Hz (scipy polyphase), never silently."""
    if sr == SR:
        return x
    try:
        from scipy.signal import resample_poly
    except ImportError:
        raise AudioIOError(f"input is {sr} Hz; resampling to {SR} Hz needs scipy "
                           f"(pip install scipy) or resample the file first") from None
    g = gcd(SR, sr)
    if notices is not None:
        notices.append(f"resampled {sr} Hz -> {SR} Hz (scipy.signal.resample_poly {SR // g}/{sr // g})")
    return resample_poly(x, SR // g, sr // g)


def quantize_pcm16(y: np.ndarray) -> np.ndarray:
    """Clip to [-1, 32767/32768] and round half-even to int16 (format §6: 16-bit PCM, [-1, 1))."""
    return np.round(np.clip(np.asarray(y, dtype=np.float64), -1.0, 32767 / 32768) * 32768.0) \
        .astype(np.int16)


def write_pcm16(path, q: np.ndarray, sr: int = SR) -> None:
    """Mono 16-bit PCM WAV with the canonical 44-byte header."""
    q = np.asarray(q)
    if q.dtype != np.int16:
        raise TypeError("write_pcm16 expects int16 samples (use quantize_pcm16)")
    try:
        with wave.open(os.fspath(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(sr))
            w.writeframes(q.astype("<i2").tobytes())
    except OSError as e:
        raise AudioIOError(f"cannot write {path}: {e}") from e
