"""Embedding (format §5–6): per-channel streams, single-frame phase write, magnitude QIM."""
from __future__ import annotations

import struct

import numpy as np

from .layout import Layout, capacity, make_layout, replicas, seed_from_pk
from .params import EPS, G, HEADER_BITS, SR, Options, Profile, WB, get_profile
from .payload import build_payload, load_secret_key, message_bytes, public_key
from .stft import analyze, polar, synthesize, wrap


def as_float_audio(x) -> np.ndarray:
    """1-D float64 samples in [-1, 1]; int16 is scaled by 1/32768."""
    a = np.asarray(x)
    if a.ndim != 1:
        raise ValueError(f"expected mono 1-D samples, got shape {a.shape} (downmix first)")
    if a.dtype == np.int16:
        return a.astype(np.float64) / 32768.0
    if not np.issubdtype(a.dtype, np.floating):
        raise TypeError(f"unsupported sample dtype {a.dtype} (use float or int16)")
    return a.astype(np.float64, copy=False)


def check_sr(sr: int) -> None:
    if int(sr) != SR:
        raise ValueError(f"sample rate must be {SR} Hz (got {sr}); resample first")


def bits_msb(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8))


def header_bits(nbits: int) -> np.ndarray:
    return bits_msb(struct.pack(">I", nbits))


def build_stream(payload: bytes, cap: int, rmax: int) -> np.ndarray:
    """h h h || b^r (format §5); ValueError if the channel cannot hold one body copy."""
    b = bits_msb(payload)
    if cap - HEADER_BITS < len(b):
        raise ValueError(f"too short: channel capacity {cap} bits < {HEADER_BITS + len(b)} needed")
    r = replicas(cap, len(b), rmax)
    return np.concatenate([np.tile(header_bits(len(b)), 3), np.tile(b, r)])


def _group_mean(v: np.ndarray, f0: np.ndarray, k: np.ndarray, fn=None) -> np.ndarray:
    """Mean over the G frames f0..f0+G-1 of v[f, k] (or fn(v[f, k])), summed left to right."""
    acc = v[f0, k] if fn is None else fn(v[f0, k])
    for j in range(1, G):
        acc = acc + (v[f0 + j, k] if fn is None else fn(v[f0 + j, k]))
    return acc / G


def _log_eps(a):
    return np.log(a + EPS)


def embed_bits(x, phase_bits, mag_bits, layout: Layout, profile: Profile,
               options: Options = Options()) -> np.ndarray:
    """Write stream positions 0..len-1 of each channel into x. Both channels read the
    original STFT; untouched cells are resynthesised from their own magnitude and phase."""
    x = as_float_audio(x)
    z = analyze(x)
    a, p = np.abs(z), np.angle(z)
    a2, p2 = a.copy(), p.copy()

    pb = np.asarray(phase_bits, dtype=np.int64)
    i = np.arange(len(pb))
    k, f = layout.kp[i % profile.bp], (i // profile.bp) * G
    if len(pb) and f[-1] >= z.shape[0]:
        raise ValueError("phase stream longer than capacity")
    delta = np.where(pb == 1, np.pi / 2, -np.pi / 2) - p[f, k]
    for j in range(options.phase_frames):
        p2[f + j, k] = p[f + j, k] + delta

    mb = np.asarray(mag_bits, dtype=np.int64)
    i = np.arange(len(mb))
    s = i % profile.bm
    k1, k2, f = layout.pairs[s, 0], layout.pairs[s, 1], (i // profile.bm) * G
    if len(mb) and f[-1] + G > z.shape[0]:
        raise ValueError("magnitude stream longer than capacity")
    dlt = float(profile.delta)
    d = _group_mean(a, f, k1, _log_eps) - _group_mean(a, f, k2, _log_eps)
    c = np.round(d / dlt).astype(np.int64)
    flip = (c & 1) != mb
    c = np.where(flip, np.where(d >= c * dlt, c + 1, c - 1), c)
    shift = c * dlt - d
    g1, g2 = np.exp(shift / 2.0), np.exp(-shift / 2.0)
    for j in range(G):
        a2[f + j, k1] = a[f + j, k1] * g1
        a2[f + j, k2] = a[f + j, k2] * g2

    return synthesize(polar(a2, wrap(p2)), x, options.tail)


def embed_payload(x, payload: bytes, seed: int, profile: Profile = WB,
                  options: Options = Options(), layout: Layout | None = None) -> np.ndarray:
    x = as_float_audio(x)
    profile = get_profile(profile)
    cap = capacity(len(x), profile)
    ps = build_stream(payload, cap.cap_phase, profile.rp)
    ms = build_stream(payload, cap.cap_mag, profile.rm)
    return embed_bits(x, ps, ms, layout if layout is not None else make_layout(seed, profile),
                      profile, options)


def sign(x, sr: int, sk, message: bytes | str, profile: Profile | str = WB,
         options: Options = Options()) -> np.ndarray:
    """Watermark `x` (mono, 44.1 kHz) with Ed25519-signed `message`. Returns float64 samples of
    the same length (not clipped or quantized)."""
    check_sr(sr)
    x = as_float_audio(x)
    profile = get_profile(profile)
    sk32 = load_secret_key(sk)
    m = message_bytes(message)
    if len(m) > options.max_msg_len:
        raise ValueError(f"message too long: {len(m)} bytes > max_msg_len {options.max_msg_len}")
    payload = build_payload(m, sk32)
    need = HEADER_BITS + 8 * len(payload)
    cap = capacity(len(x), profile)
    if min(cap.cap_phase, cap.cap_mag) < need:
        raise ValueError(
            f"too short: {len(x)} samples give {cap.groups} groups; profile {profile.name} needs "
            f"{need} bits per channel (phase cap {cap.cap_phase}, magnitude cap {cap.cap_mag})")
    seed = options.seed if options.seed is not None else seed_from_pk(public_key(sk32))
    return embed_payload(x, payload, seed, profile, options)
