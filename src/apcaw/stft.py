"""Frame analysis / synthesis (format §1): N = hop = 2048, rectangular window, no centering.

Arrays are frame-major, shape (T, N/2 + 1). Synthesis is the per-frame inverse rfft,
concatenated; samples after the last full frame follow `tail` ("passthrough" copies the
input, "zero" reproduces the legacy pad)."""
from __future__ import annotations

import numpy as np

from .layout import n_frames
from .params import N_FFT


def analyze(x: np.ndarray) -> np.ndarray:
    t = n_frames(len(x))
    return np.fft.rfft(x[: t * N_FFT].reshape(t, N_FFT), axis=1)


def wrap(p: np.ndarray) -> np.ndarray:
    """(p + pi) mod 2pi - pi with floor-mod (result in [-pi, pi))."""
    return (p + np.pi) % (2 * np.pi) - np.pi


def polar(a: np.ndarray, p: np.ndarray) -> np.ndarray:
    """a * exp(j p), written out as (a cos p, a sin p)."""
    z = np.empty(a.shape, dtype=np.complex128)
    z.real = a * np.cos(p)
    z.imag = a * np.sin(p)
    return z


def synthesize(z: np.ndarray, x: np.ndarray, tail: str) -> np.ndarray:
    t = z.shape[0]
    y = np.empty(len(x), dtype=np.float64)
    y[: t * N_FFT] = np.fft.irfft(z, n=N_FFT, axis=1).reshape(-1)
    y[t * N_FFT:] = x[t * N_FFT:] if tail == "passthrough" else 0.0
    return y
