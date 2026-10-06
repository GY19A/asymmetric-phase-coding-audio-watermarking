"""Constants, profiles (format §3) and options (switches between v1 and the legacy configuration)."""
from __future__ import annotations

from dataclasses import dataclass

SR = 44100
N_FFT = 2048          # frame length = hop (rectangular window, center=False)
G = 8                 # frames per group
HEADER_BITS = 96      # 3 copies of a 32-bit length
EPS = 1e-8            # log-magnitude floor
RS_NSYM = 30
SIG_LEN = 64
MAX_MSG_LEN = 159     # 2 + 159 + 64 = 225 bytes: one RS(255) codeword


@dataclass(frozen=True)
class Profile:
    name: str
    p_lo: int
    p_hi: int
    m_lo: int
    m_hi: int
    delta: float = 1.0
    rp: int = 1
    rm: int = 5

    @property
    def bp(self) -> int:
        """Phase bits per group."""
        return self.p_hi - self.p_lo

    @property
    def w(self) -> int:
        """Magnitude bins used (even)."""
        n = self.m_hi - self.m_lo
        return n - n % 2

    @property
    def bm(self) -> int:
        """Magnitude bits per group."""
        return self.w // 2


WB = Profile("wb", 60, 300, 100, 340)
NB = Profile("nb", 60, 300, 16, 168)
PROFILES = {"wb": WB, "nb": NB}


def get_profile(p: Profile | str) -> Profile:
    if isinstance(p, Profile):
        return p
    try:
        return PROFILES[str(p).lower()]
    except KeyError:
        raise ValueError(f"unknown profile {p!r} (expected one of {sorted(PROFILES)})") from None


@dataclass(frozen=True)
class Options:
    """v1 defaults; `Options.legacy()` reproduces the original HybridCoder configuration."""
    phase_frames: int = 1              # legacy 8: phase delta written to all G frames of a group
    verifier: str = "search"           # legacy "header": fast path only
    tail: str = "passthrough"          # legacy "zero": samples after the last full frame zeroed
    erasure_tau: float = 1e-6          # legacy 0.0: no silent-bin erasure
    seed: int | None = None            # None: derived from the public key; legacy used 42
    max_msg_len: int = MAX_MSG_LEN

    def __post_init__(self):
        if not 1 <= self.phase_frames <= G:
            raise ValueError(f"phase_frames must be in 1..{G}")
        if self.verifier not in ("search", "header"):
            raise ValueError("verifier must be 'search' or 'header'")
        if self.tail not in ("passthrough", "zero"):
            raise ValueError("tail must be 'passthrough' or 'zero'")
        if not self.erasure_tau >= 0.0:
            raise ValueError("erasure_tau must be >= 0")
        if self.seed is not None and not 0 <= int(self.seed) < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        if not 0 <= self.max_msg_len <= MAX_MSG_LEN:
            raise ValueError(f"max_msg_len must be in 0..{MAX_MSG_LEN}")

    @classmethod
    def legacy(cls) -> "Options":
        return cls(phase_frames=8, verifier="header", tail="zero", erasure_tau=0.0, seed=42)
