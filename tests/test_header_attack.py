"""Test 6: flipping every header bit on both channels defeats the legacy (header) verifier only."""
import numpy as np
import pytest

from apcaw import Options, WB, keygen, sign, verify
from apcaw.embed import build_stream, embed_bits
from apcaw.layout import capacity, make_layout, seed_from_pk
from apcaw.payload import build_payload

from conftest import FIXED_SECRET, NIPS_MSG, pem_pair

SK, PK = keygen(FIXED_SECRET)


def header_attack(y, payload, profile=WB):
    """Re-modulate stream positions 0..95 (the 3 header copies) with inverted bits, on the
    signed file itself; every other cell is analysed and resynthesised unchanged."""
    seed = seed_from_pk(PK)
    cap = capacity(len(y), profile)
    hp = build_stream(payload, cap.cap_phase, profile.rp)[:96]
    hm = build_stream(payload, cap.cap_mag, profile.rm)[:96]
    assert np.array_equal(hp, hm)
    return embed_bits(y, 1 - hp, 1 - hm, make_layout(seed, profile), profile, Options())


@pytest.fixture(scope="module")
def attacked(clips):
    payload = build_payload(NIPS_MSG, SK)
    out = []
    for x in clips:
        y = sign(x, 44100, SK, NIPS_MSG)
        out.append((y, header_attack(y, payload)))
    return out


def test_attack_really_flips_the_header(attacked):
    """Phase: every majority header bit flips. Magnitude: the header no longer reads the true
    length. (Clip 0000 has 5 digitally silent frames in group 0, where the group-mean QIM cannot
    reach the lattice, so its magnitude header is garbage rather than the exact complement.)"""
    from apcaw.verify import header_length, read_soft
    seed = seed_from_pk(PK)
    ell = 8 * len(build_payload(NIPS_MSG, SK))
    for y, ya in attacked:
        before, after = read_soft(y, seed, WB, Options()), read_soft(ya, seed, WB, Options())
        for ch, b, a in zip(("phase", "magnitude"), before, after):
            hb = b[:96].reshape(3, 32).sum(0) > 0
            ha = a[:96].reshape(3, 32).sum(0) > 0
            print(f"{ch}: header bits flipped {int(np.sum(hb != ha))}/32, "
                  f"length read {header_length(b)} -> {header_length(a)}")
            if ch == "phase":
                assert np.all(hb != ha)
            assert header_length(a) != ell
            # body cells untouched by the attack
            np.testing.assert_allclose(a[96:], b[96:], atol=1e-6)


def test_v1_search_verifies(attacked):
    for _, ya in attacked:
        r = verify(ya, 44100, PK)
        print("header attack v1:", r)
        assert r.verified and r.path == "search" and r.message == NIPS_MSG.encode()
        assert r.channel == "phase"


def test_header_verifier_fails(attacked):
    for _, ya in attacked:
        assert not verify(ya, 44100, PK, options=Options(verifier="header")).verified


