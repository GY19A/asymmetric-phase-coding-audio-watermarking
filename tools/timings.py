#!/usr/bin/env python3
"""Sign / verify wall-clock timings on the conformance clip vectors/clip_A.wav (10 s).

    .venv/bin/python tools/timings.py
"""
import sys

sys.dont_write_bytecode = True

import pathlib
import statistics
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apcaw import keygen, sign, verify                      # noqa: E402
from apcaw.io import quantize_pcm16, read_audio             # noqa: E402

VECTORS = ROOT / "vectors"
NIPS_MSG = "NIPS2026: Authenticity Token for Deepfake Defense"


def timed(f, n):
    """(median s, min s, last result) over n calls."""
    ts, r = [], None
    for _ in range(n):
        t0 = time.perf_counter()
        r = f()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), r


def pad_delay(y, s):
    out = np.zeros_like(y)
    out[s:] = y[: len(y) - s]
    return out


def main() -> int:
    x, sr = read_audio(VECTORS / "clip_A.wav")
    sk, pk = keygen(bytes(range(32)))
    _, pk_other = keygen(bytes(range(1, 33)))
    sign(x, sr, sk, "warm-up")
    verify(x[:sr], sr, pk)

    med, lo, y = timed(lambda: sign(x, sr, sk, NIPS_MSG), 20)
    print(f"clip_A: {len(x)} samples ({len(x) / sr:.2f} s)")
    print(f"{'sign':23s} median {med * 1e3:8.1f} ms  min {lo * 1e3:8.1f} ms  (n=20)")
    yq = quantize_pcm16(y).astype(np.float64) / 32768.0
    cases = [("verify clean (PCM16)", lambda: verify(yq, sr, pk), 20),
             ("verify wrong key", lambda: verify(yq, sr, pk_other), 5),
             ("verify unsigned", lambda: verify(x, sr, pk), 5),
             ("verify clean --resync", lambda: verify(yq, sr, pk, resync=True), 20),
             ("resync delay 5000", lambda: verify(pad_delay(yq, 5000), sr, pk, resync=True), 3),
             ("resync delay 6144", lambda: verify(pad_delay(yq, 6144), sr, pk, resync=True), 3),
             ("resync unsigned", lambda: verify(x, sr, pk, resync=True), 1),
             ("resync wrong key", lambda: verify(yq, sr, pk_other, resync=True), 1)]
    for name, f, n in cases:
        med, lo, r = timed(f, n)
        print(f"{name:23s} median {med * 1e3:8.1f} ms  min {lo * 1e3:8.1f} ms  (n={n})  "
              f"verified={r.verified} path={r.path} tried={r.candidates_tried} "
              f"rs={r.rs_passes} sig={r.sig_checks}  {r.reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
