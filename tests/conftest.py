"""Shared fixtures: the LibriSpeech clips, the conformance vectors and ffmpeg."""
from __future__ import annotations

import importlib
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tests" / "media"
VECTORS = ROOT / "vectors"
CLIP_NAMES = [
    "0000_librispeech_5639-40744-0030.wav",
    "0001_librispeech_8555-284447-0010.wav",
    "0002_librispeech_3570-5695-0012.wav",
]
SR = 44100
NIPS_MSG = "NIPS2026: Authenticity Token for Deepfake Defense"
FIXED_SECRET = bytes(range(32))
OTHER_SECRET = bytes(range(1, 33))

# every temporary file (pytest tmp_path, codec round trips) stays inside the repository
LOCAL_TMP = ROOT / ".tmp"
LOCAL_TMP.mkdir(exist_ok=True)
tempfile.tempdir = str(LOCAL_TMP)


def pytest_configure(config):
    if not config.option.basetemp:
        config.option.basetemp = str(LOCAL_TMP / "pytest")



def read_pcm16(path) -> np.ndarray:
    import wave
    with wave.open(str(path), "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0


def load_clip(i: int) -> np.ndarray:
    p = DATA_DIR / CLIP_NAMES[i]
    if not p.exists():
        pytest.fail(f"test clip missing: {p}")
    return read_pcm16(p)


@pytest.fixture(scope="session")
def clips():
    return [load_clip(i) for i in range(3)]


@pytest.fixture(scope="session")
def clip0():
    return load_clip(0)


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def quantize(y: np.ndarray) -> np.ndarray:
    """PCM16 quantizer used by the tools: clip to [-1, 32767/32768], round half-even."""
    return np.round(np.clip(y, -1.0, 32767.0 / 32768.0) * 32768.0).astype(np.int16)


def write_pcm16(path, y):
    import wave
    q = quantize(y)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(q.astype("<i2").tobytes())


def ffmpeg_roundtrip(y: np.ndarray, ext: str, args: list[str]) -> np.ndarray:
    """benchmark.py::_ffmpeg_tc: PCM16 wav -> codec -> 44.1 kHz mono wav, pad/trim to length."""
    if not have_ffmpeg():
        pytest.fail("ffmpeg not on PATH")
    with tempfile.TemporaryDirectory() as td:
        inp = pathlib.Path(td) / "in.wav"
        mid = pathlib.Path(td) / f"mid.{ext}"
        out = pathlib.Path(td) / "out.wav"
        write_pcm16(inp, y)
        subprocess.run(["ffmpeg", "-y", "-i", str(inp)] + args + [str(mid)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["ffmpeg", "-y", "-i", str(mid), "-ar", str(SR), "-ac", "1",
                        "-c:a", "pcm_s16le", str(out)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deg = read_pcm16(out)
    if len(deg) < len(y):
        deg = np.pad(deg, (0, len(y) - len(deg)))
    return deg[: len(y)]


def mp3_128(y):
    return ffmpeg_roundtrip(y, "mp3", ["-b:a", "128k"])


def ogg_128(y):
    return ffmpeg_roundtrip(y, "ogg", ["-c:a", "libvorbis", "-b:a", "128k"])


def crop_head(y, pct):
    """common/attacks.py::atk_crop_pct(mode='start'): zero the first pct %, keep length."""
    out = y.copy()
    out[: int(round(len(y) * pct / 100.0))] = 0.0
    return out


def crop_tail(y, pct):
    """benchmark.py::atk_crop: keep the first int(n*(1-pct/100)) samples, zero the rest."""
    cl = int(len(y) * (1 - pct / 100.0))
    out = np.zeros_like(y)
    out[:cl] = y[:cl]
    return out


def pem_pair(secret32: bytes) -> tuple[bytes, bytes]:
    """(PKCS8 private PEM, SPKI public PEM) for a raw Ed25519 seed, for the PEM key-file tests."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519
    sk = ed25519.Ed25519PrivateKey.from_private_bytes(secret32)
    priv = sk.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption())
    pub = sk.public_key().public_bytes(serialization.Encoding.PEM,
                                       serialization.PublicFormat.SubjectPublicKeyInfo)
    return priv, pub
