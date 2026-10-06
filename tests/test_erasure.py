"""Test 7: silent-bin erasure. Cells written into digital silence read as exactly 0 (v1),
instead of the legacy confident -1 (tau = 0)."""
import numpy as np
import pytest

from apcaw import Options, WB, keygen, sign, verify
from apcaw.embed import build_stream
from apcaw.layout import capacity, seed_from_pk
from apcaw.payload import build_payload
from apcaw.verify import combine_body, read_soft

from conftest import FIXED_SECRET, NIPS_MSG

SK, PK = keygen(FIXED_SECRET)
LEAD = 3 * 44100                               # 3 s of digital silence before the speech
SILENT_GROUPS = LEAD // (2048 * 8)             # groups whose 8 frames lie entirely in the zeros


@pytest.fixture(scope="module")
def padded(clips):
    out = []
    for x in clips:
        xp = np.concatenate([np.zeros(LEAD), x])
        out.append(sign(xp, 44100, SK, NIPS_MSG))
    return out


def test_output_is_exactly_zero_in_silence(padded):
    for y in padded:
        assert not np.any(y[: SILENT_GROUPS * 8 * 2048])


def test_silent_positions_are_erased(padded):
    seed = seed_from_pk(PK)
    for y in padded:
        ps1, ms1 = read_soft(y, seed, WB, Options())
        ps0, ms0 = read_soft(y, seed, WB, Options(erasure_tau=0.0))
        nm, np_ = SILENT_GROUPS * WB.bm, SILENT_GROUPS * WB.bp
        assert np.all(ms1[:nm] == 0.0)
        assert np.all(ms0[:nm] == -1.0)                  # the legacy bias: confident bit 0
        assert np.all(ps1[:np_] == 0.0) and np.all(ps0[:np_] == 0.0)
        # outside the silence nothing changes between the two readers on this material
        assert np.count_nonzero(ms1[nm:] != ms0[nm:]) <= np.count_nonzero(ms1[nm:] == 0.0)


def test_erasure_never_increases_body_bit_errors(padded):
    seed = seed_from_pk(PK)
    payload = build_payload(NIPS_MSG, SK)
    truth = np.unpackbits(np.frombuffer(payload, np.uint8))
    for y in padded:
        cap = capacity(len(y), WB)
        r = len(build_stream(payload, cap.cap_mag, WB.rm)) - 96
        r //= len(truth)
        _, ms1 = read_soft(y, seed, WB, Options())
        _, ms0 = read_soft(y, seed, WB, Options(erasure_tau=0.0))
        e1 = int(np.sum((combine_body(ms1, len(truth), r) > 0) != truth))
        e0 = int(np.sum((combine_body(ms0, len(truth), r) > 0) != truth))
        print(f"mag replicas={r} body bit errors: v1={e1} tau0={e0}")
        assert e1 <= e0


def test_v1_verifies_despite_silent_header(padded):
    for y in padded:
        r = verify(y, 44100, PK)
        print("3 s lead silence:", r)
        assert r.verified and r.channel == "magnitude" and r.path == "search"
        assert not verify(y, 44100, PK, options=Options(verifier="header")).verified
