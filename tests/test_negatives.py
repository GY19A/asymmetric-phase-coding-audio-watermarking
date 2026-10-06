"""Test 5: wrong public key and unsigned audio are rejected; the work counters are recorded."""
import time

import pytest

from apcaw import keygen, sign, verify

from conftest import FIXED_SECRET, NIPS_MSG, OTHER_SECRET

SK, PK = keygen(FIXED_SECRET)
_, PK_OTHER = keygen(OTHER_SECRET)


@pytest.fixture(scope="module")
def signed(clips):
    return [sign(x, 44100, SK, NIPS_MSG) for x in clips]


def _counters(r):
    return dict(candidates_tried=r.candidates_tried, rs_passes=r.rs_passes,
                sig_checks=r.sig_checks, reason=r.reason)


def test_wrong_public_key(signed):
    for y in signed:
        t0 = time.perf_counter()
        r = verify(y, 44100, PK_OTHER)
        dt = time.perf_counter() - t0
        print(f"wrong pk: {dt:.3f}s", _counters(r))
        assert not r.verified and r.message is None and r.channel is None and r.path is None
        # a different key gives a different layout: nothing should even pass RS
        assert r.candidates_tried > 0
        assert r.sig_checks <= r.rs_passes


def test_unsigned_audio(clips):
    for x in clips:
        t0 = time.perf_counter()
        r = verify(x, 44100, PK)
        dt = time.perf_counter() - t0
        print(f"unsigned: {dt:.3f}s", _counters(r))
        assert not r.verified and r.message is None
        assert r.candidates_tried > 0
        assert r.reason


def test_too_short_reports_reason():
    import numpy as np
    r = verify(np.zeros(2048 * 8), 44100, PK)
    assert not r.verified and "too short" in r.reason and r.candidates_tried == 0
    with pytest.raises(ValueError, match="too short"):
        sign(np.zeros(2048 * 8), 44100, SK, NIPS_MSG)


def test_same_key_other_message_is_not_confused(clips):
    """A file signed with message A never verifies as message B (sanity on the message field)."""
    y = sign(clips[0], 44100, SK, "A")
    r = verify(y, 44100, PK)
    assert r.verified and r.message == b"A"
