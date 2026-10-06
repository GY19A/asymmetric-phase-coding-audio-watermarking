"""Built-in self-test: a minimal copy of the format/vectors conformance set (layout, MT19937,
payload / stream, Reed-Solomon) plus one synthetic signing, written by tools/make_vectors.py into
apcaw/data/selftest.json. No audio files are bundled; the signing check uses a portable
MT19937-generated signal."""
from __future__ import annotations

import json
from importlib import resources

import numpy as np

from . import layout as _layout
from .embed import build_stream, sign
from .params import WB, get_profile
from .payload import build_payload, keygen, rs_decode
from .verify import verify

SIGNAL_SEED = 20260928
SIGNAL_LEN = 131072                         # 64 frames = 8 groups
SAMPLE_STRIDE = 509                         # ~258 sampled output values
SECRET_SEED = bytes(range(32))
TOL = 1e-9


def signal(seed: int = SIGNAL_SEED, n: int = SIGNAL_LEN) -> np.ndarray:
    """Portable test signal: x[i] = (u_i / 2^32 - 0.5) / 2 with u_i the MT19937 outputs of
    init_genrand(seed) (exact in float64)."""
    u = np.random.RandomState(seed).randint(0, 2**32, size=n, dtype=np.uint64)
    return (u.astype(np.float64) / 2.0**32 - 0.5) * 0.5


def build(vectors: dict) -> dict:
    """`vectors`: the parsed layout / mt19937 / payload / rs JSON files of vectors/."""
    sk, pk = keygen(SECRET_SEED)
    y = sign(signal(), 44100, sk, "A")
    r = verify(y, 44100, pk)
    idx = list(range(0, SIGNAL_LEN, SAMPLE_STRIDE))
    return {
        "format": "apcaw-v1 selftest",
        "vectors": vectors,
        "signal": {"seed": SIGNAL_SEED, "n": SIGNAL_LEN, "message": "A", "profile": "wb",
                   "secret_seed_hex": SECRET_SEED.hex(),
                   "sample_index": idx, "sample_value": [float(y[i]) for i in idx]},
        "verify": {"verified": r.verified, "channel": r.channel, "path": r.path,
                   "profile": r.profile, "rs_corrected": r.rs_corrected,
                   "message_hex": r.message.hex() if r.message else None},
    }


def load() -> dict:
    return json.loads(resources.files("apcaw").joinpath("data/selftest.json").read_text())


def run(ref: dict | None = None) -> list[tuple[str, bool, str]]:
    """[(check, passed, detail)]."""
    ref = ref if ref is not None else load()
    vec = ref["vectors"]
    out = []

    def check(name, ok, detail=""):
        out.append((name, bool(ok), detail))

    pv = vec["payload"]
    sk, pk = keygen(bytes.fromhex(pv["secret_seed_hex"]))
    check("keygen", pk.hex() == pv["public_key_hex"])
    check("seed_from_pk", _layout.seed_from_pk(pk) == pv["key_seed"])
    mt = vec["mt19937"]["seeds"]
    check("mt19937", all(_layout.mt19937_outputs(e["seed"], len(e["outputs"])) == e["outputs"]
                         for e in mt), f"{len(mt)} seeds")
    u = np.array(_layout.mt19937_outputs(ref["signal"]["seed"], 8), dtype=np.float64)
    check("signal", np.array_equal(signal(ref["signal"]["seed"], 8), (u / 2.0**32 - 0.5) * 0.5))

    ok, n = True, 0
    for e in vec["layout"]["seeds"]:
        for name in ("wb", "nb"):
            p = get_profile(name)
            ok &= _layout.phase_bins(e["seed"], p).tolist() == e[name]["phase_bins"]
            ok &= _layout.mag_pairs(e["seed"], p).tolist() == e[name]["mag_pairs"]
            ok &= _layout.py_shuffle(e["seed"], list(range(p.p_lo, p.p_hi))) == e[name]["phase_bins"]
            n += 1
    check("layout", ok, f"{n} seed/profile pairs")

    ok_p = ok_s = True
    for e in pv["messages"]:
        m = bytes.fromhex(e["message_hex"])
        p = build_payload(m, sk)
        ok_p &= p.hex() == e["payload_hex"] and p[2 + len(m):-30].hex() == e["signature_hex"]
        for ch, st in e["streams_clip_A_wb"].items():
            rmax = WB.rp if ch == "phase" else WB.rm
            s = build_stream(p, st["capacity"], rmax)
            ok_s &= len(s) == st["stream_len"] and np.packbits(s).tobytes().hex() == st["stream_hex"]
    check("payload", ok_p, f"{len(pv['messages'])} messages")
    check("stream", ok_s)

    ok, n = True, 0
    for e in vec["rs"]["cases"]:
        for c in e["corrupted"]:
            d = rs_decode(bytes.fromhex(c["received_hex"]))
            got = (False, None, None) if d is None else (True, d[0].hex(), d[1])
            ok &= got == (c["decodes"], c["decoded_hex"], c["n_corrected"])
            n += 1
    check("reed-solomon", ok, f"{n} corrupted codewords")

    s = ref["signal"]
    sk_s, pk_s = keygen(bytes.fromhex(s["secret_seed_hex"]))
    y = sign(signal(s["seed"], s["n"]), 44100, sk_s, s["message"], get_profile(s["profile"]))
    err = float(np.max(np.abs(y[s["sample_index"]] - np.array(s["sample_value"]))))
    check("sign", err <= TOL, f"max abs diff {err:.3g} (tol {TOL:g})")
    r = verify(y, 44100, pk_s)
    v = ref["verify"]
    check("verify", (r.verified, r.channel, r.path, r.profile, r.rs_corrected,
                     r.message.hex() if r.message else None) ==
          (v["verified"], v["channel"], v["path"], v["profile"], v["rs_corrected"], v["message_hex"]))
    _, pk2 = keygen(bytes(range(1, 33)))
    check("verify-wrong-key", not verify(y, 44100, pk2).verified)
    return out
