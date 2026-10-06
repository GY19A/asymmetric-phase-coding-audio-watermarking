"""apcaw: reference implementation of APC audio watermarking, format apcaw-v1."""
__version__ = "1.0.0"

from .embed import build_stream, embed_payload, sign
from .layout import capacity, mag_pairs, make_layout, phase_bins, seed_from_pk
from .params import NB, WB, Options, Profile
from .payload import build_payload, keygen, parse_payload
from .verify import VerifyResult, extract_payload_candidates, read_soft, verify

__all__ = [
    "__version__", "keygen", "sign", "verify", "Options", "Profile", "WB", "NB", "VerifyResult",
    "build_payload", "parse_payload", "phase_bins", "mag_pairs", "make_layout", "seed_from_pk",
    "capacity", "build_stream", "embed_payload", "read_soft", "extract_payload_candidates",
]
