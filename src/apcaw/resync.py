"""Optional decoder-side resync (format §8), following a reference decoder:
score every integer offset δ ∈ [0, N) and whole-frame skip f0 ∈ [0, FMAX] by the
magnitude-weighted phase alignment Σ|Im Z| / Σ|Z| over the phase cells, then run §8 steps 1-3
on δ = 0 (every f0) and on the 8 best-scoring δ. Ed25519 still gates every acceptance."""
from __future__ import annotations

import numpy as np

from .params import G, N_FFT

FMAX = 10
SCORE_GROUPS = 6
TOPK_DELTA = 8
SCORE_FRAMES = FMAX + (SCORE_GROUPS - 1) * G + 1        # 51


def score_grid(x: np.ndarray, p_lo: int, p_hi: int) -> np.ndarray:
    """S[δ, f0]; -1 where the frames f0 + g·G (g < SCORE_GROUPS) do not all exist or carry no
    energy. The phase bin set Kp is a permutation of p_lo..p_hi-1, so the score is key-free.
    Ranking only: the value is not part of the bit-exact contract."""
    s = np.full((N_FFT, FMAX + 1), -1.0)
    fidx = np.arange(FMAX + 1)[:, None] + G * np.arange(SCORE_GROUPS)[None, :]
    for d in range(N_FFT):
        nf = min(SCORE_FRAMES, max(0, (len(x) - d) // N_FFT))
        if nf == 0:
            continue
        z = np.fft.rfft(x[d:d + nf * N_FFT].reshape(nf, N_FFT), axis=1)[:, p_lo:p_hi]
        num, den = np.abs(z.imag).sum(axis=1), np.abs(z).sum(axis=1)
        ok = fidx[:, -1] < nf
        n, dd = num[fidx[ok]].sum(axis=1), den[fidx[ok]].sum(axis=1)
        s[d, ok] = np.where(dd > 0, n / np.where(dd > 0, dd, 1.0), -1.0)
    return s


def resync_search(x: np.ndarray, st):
    """Try alignments in reference order on the verify state `st` (shared de-duplication and
    counters). (δ, f0) = (0, 0) is the plain path and is skipped."""
    from .stft import analyze
    p_lo, p_hi = st.profiles[0][0].p_lo, st.profiles[0][0].p_hi
    s = score_grid(x, p_lo, p_hi)
    best = s.max(axis=1)
    order = [0] + [int(i) for i in np.argsort(-best, kind="stable")[:TOPK_DELTA] if i != 0]
    for d in order:
        z = analyze(x[d:])
        if z.shape[0] == 0:
            continue
        for f0 in np.argsort(-s[d], kind="stable"):
            f0 = int(f0)
            if s[d, f0] < 0 or (d, f0) == (0, 0):
                continue
            r = st.run(z[f0:], path_override="resync")
            if r is not None:
                return type(r)(**{**r.__dict__, "reason": f"ok (resync delta={d} f0={f0})"})
    return None
