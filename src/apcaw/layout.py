"""Key-derived layout (format §2): seed, shuffled phase bins Kp, permuted magnitude pairs.

The reference uses numpy's legacy `RandomState(seed).shuffle` (MT19937 `init_genrand` +
Fisher-Yates with masked rejection sampling). `mt19937_outputs` / `py_shuffle` are dependency-free
restatements of the same algorithm for porters; the tests pin them to numpy.
"""
from __future__ import annotations

import hashlib
from typing import NamedTuple

import numpy as np

from .params import G, HEADER_BITS, N_FFT, Profile

MAG_SEED_XOR = 0xDEADBEEF


def seed_from_pk(pk) -> int:
    """seed = first 8 bytes of SHA-256(raw 32-byte public key), big endian, mod 2^32."""
    from .payload import load_public_key
    return int.from_bytes(hashlib.sha256(load_public_key(pk)).digest()[:8], "big") % 2**32


def _check_seed(seed: int) -> int:
    seed = int(seed)
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    return seed


def phase_bins(seed: int, profile: Profile) -> np.ndarray:
    freqs = np.arange(profile.p_lo, profile.p_hi)
    np.random.RandomState(_check_seed(seed)).shuffle(freqs)
    return freqs


def mag_pairs(seed: int, profile: Profile) -> np.ndarray:
    pairs = np.arange(profile.m_lo, profile.m_lo + profile.w).reshape(-1, 2)
    perm = np.arange(len(pairs))
    np.random.RandomState((_check_seed(seed) ^ MAG_SEED_XOR) & 0xFFFFFFFF).shuffle(perm)
    return pairs[perm]


class Layout(NamedTuple):
    kp: np.ndarray        # (Bp,) phase bin per slot
    pairs: np.ndarray     # (Bm, 2) magnitude pair per slot


def make_layout(seed: int, profile: Profile) -> Layout:
    return Layout(phase_bins(seed, profile), mag_pairs(seed, profile))


class Capacity(NamedTuple):
    frames: int
    groups: int
    cap_phase: int
    cap_mag: int


def n_frames(n_samples: int) -> int:
    return 0 if n_samples < N_FFT else (n_samples - N_FFT) // N_FFT + 1


def capacity(n_samples: int, profile: Profile) -> Capacity:
    t = n_frames(n_samples)
    g = t // G
    return Capacity(t, g, g * profile.bp, g * profile.bm)


def replicas(cap: int, nbits: int, rmax: int) -> int:
    """Format §5: r = max(1, min(R, floor((cap - 96) / |b|)))."""
    return max(1, min(int(rmax), (cap - HEADER_BITS) // nbits))


# ----------------------------------------------------------------------------------------------
# Dependency-free MT19937 / shuffle (reference for ports; identical to numpy RandomState)
# ----------------------------------------------------------------------------------------------
class _MT19937:
    def __init__(self, seed: int):
        mt = [0] * 624
        mt[0] = _check_seed(seed)
        for i in range(1, 624):
            mt[i] = (1812433253 * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i) & 0xFFFFFFFF
        self.mt, self.idx = mt, 624

    def _twist(self):
        mt = self.mt
        for i in range(624):
            y = (mt[i] & 0x80000000) | (mt[(i + 1) % 624] & 0x7FFFFFFF)
            mt[i] = mt[(i + 397) % 624] ^ (y >> 1) ^ (0x9908B0DF if y & 1 else 0)
        self.idx = 0

    def next_u32(self) -> int:
        if self.idx >= 624:
            self._twist()
        y = self.mt[self.idx]
        self.idx += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y

    def interval(self, mx: int) -> int:
        """numpy `random_interval`: uniform in [0, mx] by masked rejection."""
        if mx == 0:
            return 0
        mask = mx
        for s in (1, 2, 4, 8, 16):
            mask |= mask >> s
        while True:
            v = self.next_u32() & mask
            if v <= mx:
                return v


def mt19937_outputs(seed: int, n: int) -> list[int]:
    g = _MT19937(seed)
    return [g.next_u32() for _ in range(n)]


def py_shuffle(seed: int, items: list) -> list:
    """RandomState(seed).shuffle(items): for i = n-1 .. 1, swap items[i] with items[interval(i)]."""
    out = list(items)
    g = _MT19937(seed)
    for i in range(len(out) - 1, 0, -1):
        j = g.interval(i)
        out[i], out[j] = out[j], out[i]
    return out
