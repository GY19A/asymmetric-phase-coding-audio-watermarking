"""Regression against the frozen conformance vectors in vectors/ (format §11)."""
import hashlib
import json

import numpy as np
import pytest
from reedsolo import RSCodec, ReedSolomonError

from apcaw import NB, WB, Options, keygen, sign, verify
from apcaw.io import read_audio
from apcaw.layout import mag_pairs, mt19937_outputs, phase_bins
from apcaw.embed import build_stream
from apcaw.payload import build_payload
from apcaw.verify import read_soft
from apcaw.layout import seed_from_pk

from conftest import VECTORS


def _load(name):
    p = VECTORS / name
    if not p.exists():
        pytest.fail(f"missing vector {p} (run tools/make_vectors.py)")
    return json.loads(p.read_text())


def test_manifest_hashes():
    m = _load("MANIFEST.json")
    assert m["files"]
    for name, meta in m["files"].items():
        assert hashlib.sha256((VECTORS / name).read_bytes()).hexdigest() == meta["sha256"], name


def test_layout_and_mt():
    for e in _load("layout.json")["seeds"]:
        for prof in (WB, NB):
            assert phase_bins(e["seed"], prof).tolist() == e[prof.name]["phase_bins"]
            assert mag_pairs(e["seed"], prof).tolist() == e[prof.name]["mag_pairs"]
    for e in _load("mt19937.json")["seeds"]:
        assert mt19937_outputs(e["seed"], 16) == e["outputs"]
        assert np.random.RandomState(e["seed"]).randint(0, 2**32, 16, dtype=np.uint64).tolist() \
            == e["outputs"]


def test_payload():
    v = _load("payload.json")
    sk, pk = keygen(bytes.fromhex(v["secret_seed_hex"]))
    assert pk.hex() == v["public_key_hex"]
    for e in v["messages"]:
        p = build_payload(bytes.fromhex(e["message_hex"]), sk)
        assert p.hex() == e["payload_hex"]
        assert p[2 + len(bytes.fromhex(e["message_hex"])):-30].hex() == e["signature_hex"]
        for ch, st in e["streams_clip_A_wb"].items():
            prof_r = WB.rp if ch == "phase" else WB.rm
            b = build_stream(p, st["capacity"], prof_r)
            assert len(b) == st["stream_len"] == 96 + st["replicas"] * e["payload_bits"]
            assert np.packbits(b).tobytes().hex() == st["stream_hex"]
            assert "".join(map(str, b[:32])) == e["header_bits"]


def test_rs():
    rsc = RSCodec(30)
    for e in _load("rs.json")["cases"]:
        assert rsc.encode(bytes.fromhex(e["message_hex"])).hex() == e["codeword_hex"]
        for c in e["corrupted"]:
            try:
                msg, _, errata = rsc.decode(bytes.fromhex(c["received_hex"]))
                got = (True, bytes(msg).hex(), len(errata))
            except ReedSolomonError:
                got = (False, None, None)
            assert got == (c["decodes"], c["decoded_hex"], c["n_corrected"])


def test_signed_audio_reproduces():
    v = _load("payload.json")
    sk, pk = keygen(bytes.fromhex(v["secret_seed_hex"]))
    x, sr = read_audio(VECTORS / "clip_A.wav")
    y = sign(x, sr, sk, "NIPS2026: Authenticity Token for Deepfake Defense")
    ref = np.fromfile(VECTORS / "clip_A_signed_v1.f64", dtype="<f8")
    np.testing.assert_allclose(y, ref, atol=1e-9, rtol=0)  # format §11; bit-identical on one machine
    assert np.array_equal(np.load(VECTORS / "clip_A_signed_v1.npy"), ref)
    w, _ = read_audio(VECTORS / "clip_A_signed_v1.wav")
    assert np.array_equal(w, np.round(np.clip(ref, -1, 32767 / 32768) * 32768) / 32768)


def test_soft_and_verify_results():
    v = _load("soft_A.json")
    _, pk = keygen(bytes.fromhex(_load("payload.json")["secret_seed_hex"]))
    seed = seed_from_pk(pk)
    for e in v["files"]:
        legacy = e["options"] == "legacy"
        o = Options.legacy() if legacy else Options()
        x, _ = read_audio(VECTORS / e["file"])
        ps, ms = read_soft(x, o.seed if legacy else seed, WB, o)
        np.testing.assert_allclose(ps, np.array(e["phase_soft"]), atol=1e-9, rtol=0)
        np.testing.assert_allclose(ms, np.array(e["mag_soft"]), atol=1e-9, rtol=0)
        r = verify(x, 44100, pk, WB if legacy else None, o)
        got = (r.verified, r.message.hex() if r.message is not None else None, r.channel,
               r.profile, r.path, r.rs_corrected, r.payload_bits,
               r.candidates_tried, r.rs_passes, r.sig_checks)
        want = tuple(e["verify"][k] for k in ("verified", "message_hex", "channel", "profile", "path",
                                              "rs_corrected", "payload_bits", "candidates_tried",
                                              "rs_passes", "sig_checks"))
        assert got == want, e["file"]
    ok = {e["file"]: e["verify"]["verified"] for e in v["files"]}
    assert ok == {"clip_A_signed_v1.wav": True, "clip_A_signed_legacy.wav": True,
                  "clip_A_signed_v1_mp3.wav": True}


def test_negatives():
    for e in _load("negatives.json")["cases"]:
        x, _ = read_audio(VECTORS / e["file"])
        r = verify(x, 44100, bytes.fromhex(e["public_key_hex"]))
        assert r.verified is False and e["verified"] is False
        assert (r.candidates_tried, r.rs_passes, r.sig_checks) == \
            (e["candidates_tried"], e["rs_passes"], e["sig_checks"])


def test_selftest_bundle_is_a_copy_of_the_vectors():
    from apcaw import selftest
    ref = selftest.load()
    for name in ("layout", "mt19937", "payload", "rs"):
        assert ref["vectors"][name] == _load(f"{name}.json"), name
    assert all(ok for _, ok, _ in selftest.run(ref))
