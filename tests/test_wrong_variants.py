"""Test 8: deliberately wrong variants must be caught.

8a  An implementation that swaps two layout bins is detected (layout vector mismatch, raw bit
    errors on the affected channel); what RS then makes of it is recorded, not assumed.
8b  A payload signed by another key, embedded on *this* key's layout, decodes through RS but is
    rejected by Ed25519.
8c  The legacy seed-42 layout is not the key-derived layout: a legacy file does not verify under
    default v1 options, only when the legacy seed is supplied.
"""
import numpy as np
import pytest

from apcaw import Options, WB, keygen, sign, verify
from apcaw.embed import build_stream, embed_payload
from apcaw.layout import Layout, capacity, make_layout, seed_from_pk
from apcaw.payload import build_payload
from apcaw.verify import read_soft

from conftest import FIXED_SECRET, NIPS_MSG, OTHER_SECRET

SK, PK = keygen(FIXED_SECRET)
SK_B, PK_B = keygen(OTHER_SECRET)
SEED = seed_from_pk(PK)


def _bit_errors(soft, stream):
    return int(np.sum((soft[: len(stream)] > 0) != stream.astype(bool)))


def test_8a_swapped_phase_bins_detected(clips):
    good = make_layout(SEED, WB)
    kp = good.kp.copy()
    kp[[0, 1]] = kp[[1, 0]]
    bad = Layout(kp, good.pairs)
    assert not np.array_equal(bad.kp, good.kp)             # the layout vector catches it
    payload = build_payload(NIPS_MSG, SK)
    for x in clips:
        y_bad = embed_payload(x, payload, SEED, WB, Options(), layout=bad)
        y_good = sign(x, 44100, SK, NIPS_MSG)
        stream = build_stream(payload, capacity(len(x), WB).cap_phase, WB.rp)
        ps, _ = read_soft(y_bad, SEED, WB, Options())
        nerr = _bit_errors(ps, stream)
        diff = float(np.max(np.abs(y_bad - y_good)))
        r = verify(y_bad, 44100, PK)
        print(f"8a swap Kp[0]<->Kp[1]: phase raw bit errors={nerr} max|y_bad-y_good|={diff:.3g} "
              f"verify={r.verified} ch={r.channel} rs_corrected={r.rs_corrected}")
        # a swap is a real defect: either it shows as raw bit errors that RS has to repair, or
        # (if both slots carry equal bits in every group) the output is bit-identical and only
        # the layout vector can catch it
        if nerr == 0:
            assert diff == 0.0
        else:
            assert diff > 1e-9
            assert (not r.verified) or r.channel != "phase" or r.rs_corrected > 0


def test_8a_swapped_mag_pairs_detected(clips):
    good = make_layout(SEED, WB)
    pairs = good.pairs.copy()
    pairs[[0, 1]] = pairs[[1, 0]]
    bad = Layout(good.kp, pairs)
    payload = build_payload(NIPS_MSG, SK)
    x = clips[0]
    y_bad = embed_payload(x, payload, SEED, WB, Options(), layout=bad)
    stream = build_stream(payload, capacity(len(x), WB).cap_mag, WB.rm)
    _, ms = read_soft(y_bad, SEED, WB, Options())
    nerr = _bit_errors(ms, stream)
    print(f"8a swap pairs[0]<->pairs[1]: mag raw bit errors={nerr}")
    assert nerr > 0


def test_8b_other_key_payload_rs_ok_signature_rejected(clips):
    for x in clips:
        y = sign(x, 44100, SK_B, NIPS_MSG, WB, Options(seed=SEED))   # B's signature, A's layout
        r = verify(y, 44100, PK)
        print("8b:", r)
        assert not r.verified and r.message is None
        assert r.rs_passes >= 1 and r.sig_checks >= 1
        assert verify(y, 44100, PK_B, options=Options(seed=SEED)).verified


def test_8c_legacy_seed_is_not_the_key_seed(clips):
    y = sign(clips[0], 44100, SK, NIPS_MSG, WB, Options.legacy())
    assert SEED != 42
    assert not verify(y, 44100, PK).verified
    assert verify(y, 44100, PK, options=Options(seed=42)).verified
