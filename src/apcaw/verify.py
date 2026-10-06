"""Soft reading (format §7) and header-independent verification (format §8)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embed import _group_mean, _log_eps, as_float_audio, check_sr
from .layout import Layout, make_layout, replicas, seed_from_pk
from .params import G, HEADER_BITS, NB, WB, Options, Profile, get_profile
from .payload import load_public_key, parse_payload
from .stft import analyze

MIN_PAYLOAD_BITS = 8 * 96                      # empty message: be16 + S64 + 30 RS symbols


@dataclass(frozen=True)
class VerifyResult:
    verified: bool
    message: bytes | None = None
    channel: str | None = None                 # "phase" | "magnitude"
    profile: str | None = None                 # profile under which decoding succeeded
    path: str | None = None                    # "header" | "search" | "resync"
    rs_corrected: int | None = None
    candidates_tried: int = 0                  # distinct candidate byte strings RS-decoded
    rs_passes: int = 0                         # of those, RS decoding succeeded
    sig_checks: int = 0                        # of those, length consistent -> Ed25519 run
    payload_bits: int | None = None            # |b| of the accepted candidate
    reason: str = ""

    def __bool__(self) -> bool:
        return self.verified


# ----------------------------------------------------------------------------------------------
# Soft reading
# ----------------------------------------------------------------------------------------------
def soft_from_spectrum(z: np.ndarray, layout: Layout, profile: Profile, tau: float):
    """(phase_soft, mag_soft) over the full capacity of spectrum z (frame-major)."""
    groups = z.shape[0] // G
    i = np.arange(groups * profile.bp)
    k, f = layout.kp[i % profile.bp], (i // profile.bp) * G
    c = z[f, k]
    ps = np.sin(np.angle(c))
    if tau > 0:
        ps[np.abs(c) < tau] = 0.0

    i = np.arange(groups * profile.bm)
    s = i % profile.bm
    k1, k2, f = layout.pairs[s, 0], layout.pairs[s, 1], (i // profile.bm) * G
    a = np.abs(z[:, profile.m_lo:profile.m_lo + profile.w])
    k1, k2 = k1 - profile.m_lo, k2 - profile.m_lo
    d = _group_mean(a, f, k1, _log_eps) - _group_mean(a, f, k2, _log_eps)
    ms = -np.cos(np.pi * d / float(profile.delta))
    if tau > 0:
        ms[(_group_mean(a, f, k1) < tau) & (_group_mean(a, f, k2) < tau)] = 0.0
    return ps, ms


def read_soft(x, seed: int, profile: Profile = WB, options: Options = Options(),
              layout: Layout | None = None):
    profile = get_profile(profile)
    layout = layout if layout is not None else make_layout(seed, profile)
    return soft_from_spectrum(analyze(as_float_audio(x)), layout, profile, options.erasure_tau)


def combine_body(soft: np.ndarray, nbits: int, r: int) -> np.ndarray:
    """Σ over replicas, summed in replica order."""
    acc = soft[HEADER_BITS:HEADER_BITS + nbits].copy()
    for j in range(1, r):
        acc += soft[HEADER_BITS + j * nbits:HEADER_BITS + (j + 1) * nbits]
    return acc


def header_length(soft: np.ndarray) -> int | None:
    """ℓ from the majority of the three 32-bit header copies (None if the stream is < 96)."""
    if len(soft) < HEADER_BITS:
        return None
    h = soft[0:32] + soft[32:64] + soft[64:96]
    return int.from_bytes(np.packbits(h > 0).tobytes(), "big")


def _bytes(bits_soft: np.ndarray) -> bytes:
    return np.packbits(bits_soft > 0).tobytes()


def extract_payload_candidates(soft: np.ndarray, rmax: int, options: Options = Options()):
    """Yield (path, nbits, candidate bytes) in §8 order for one channel's soft stream."""
    cap = len(soft)
    ell = header_length(soft)
    if options.verifier == "header":
        # legacy HybridCoder rule
        if ell is not None and ell != 0 and ell <= cap - HEADER_BITS and ell % 8 == 0:
            yield "header", ell, _bytes(combine_body(soft, ell, replicas(cap, ell, rmax)))
        return
    hi = 8 * (96 + options.max_msg_len)
    if ell is not None and ell % 8 == 0 and MIN_PAYLOAD_BITS <= ell <= hi \
            and ell <= cap - HEADER_BITS:
        yield "header", ell, _bytes(combine_body(soft, ell, replicas(cap, ell, rmax)))
    for m in range(options.max_msg_len + 1):
        nb = 8 * (m + 96)
        if cap - HEADER_BITS < nb:
            break
        yield "search", nb, _bytes(combine_body(soft, nb, replicas(cap, nb, rmax)))


# ----------------------------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------------------------
class _Search:
    """State of one verify call: global de-duplication of candidates plus work counters."""

    def __init__(self, pk32: bytes, seed: int, profiles, options: Options):
        self.pk32, self.options = pk32, options
        self.profiles = [(p, make_layout(seed, p)) for p in profiles]
        self.seen: set[bytes] = set()
        self.candidates_tried = self.rs_passes = self.sig_checks = 0
        self.any_candidate_possible = False

    def counters(self) -> dict:
        return dict(candidates_tried=self.candidates_tried, rs_passes=self.rs_passes,
                    sig_checks=self.sig_checks)

    def run(self, z: np.ndarray, path_override: str | None = None) -> VerifyResult | None:
        """§8 steps 1-3 on spectrum z for every profile; first acceptance wins."""
        for profile, layout in self.profiles:
            ps, ms = soft_from_spectrum(z, layout, profile, self.options.erasure_tau)
            for channel, soft, rmax in (("phase", ps, profile.rp), ("magnitude", ms, profile.rm)):
                if len(soft) - HEADER_BITS >= MIN_PAYLOAD_BITS:
                    self.any_candidate_possible = True
                for path, nbits, cand in extract_payload_candidates(soft, rmax, self.options):
                    if cand in self.seen:
                        continue
                    self.seen.add(cand)
                    self.candidates_tried += 1
                    res = parse_payload(cand, self.pk32)
                    self.rs_passes += res.rs_ok
                    self.sig_checks += res.sig_checked
                    if res.ok:
                        return VerifyResult(
                            True, res.message, channel, profile.name, path_override or path,
                            res.rs_corrected, payload_bits=nbits, reason="ok", **self.counters())
        return None

    def failure(self, n_samples: int) -> VerifyResult:
        if self.candidates_tried == 0:
            if not self.any_candidate_possible:
                reason = (f"too short: {n_samples} samples cannot carry the smallest payload "
                          f"({HEADER_BITS + MIN_PAYLOAD_BITS} bits per channel)")
            else:
                reason = "no candidate: header length invalid and search disabled"
        elif self.sig_checks:
            reason = (f"signature invalid ({self.sig_checks} candidate(s) passed RS with a "
                      f"consistent length; none verified under this public key)")
        elif self.rs_passes:
            reason = f"length field inconsistent ({self.rs_passes} RS-decodable candidate(s))"
        else:
            reason = f"no candidate passed RS decoding ({self.candidates_tried} tried)"
        return VerifyResult(False, reason=reason, **self.counters())


def verify(x, sr: int, pk, profile: Profile | str | None = None, options: Options = Options(),
           resync: bool = False) -> VerifyResult:
    """Format §8. `profile=None` tries wb then nb. `resync=True` adds the §8 offset search, run
    only when the plain alignment does not verify."""
    check_sr(sr)
    x = as_float_audio(x)
    pk32 = load_public_key(pk)
    profiles = [get_profile(profile)] if profile is not None else [WB, NB]
    seed = options.seed if options.seed is not None else seed_from_pk(pk32)
    st = _Search(pk32, seed, profiles, options)
    r = st.run(analyze(x))
    if r is not None:
        return r
    if resync:
        from .resync import resync_search
        r = resync_search(x, st)
        if r is not None:
            return r
    return st.failure(len(x))
