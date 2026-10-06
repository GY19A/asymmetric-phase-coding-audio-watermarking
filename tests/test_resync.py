"""Optional decoder-side resync (format §8, off by default) on a pure integer-sample delay."""
import time

import numpy as np
import pytest

from apcaw import keygen, sign, verify

from conftest import FIXED_SECRET, NIPS_MSG

SK, PK = keygen(FIXED_SECRET)


def pad_delay(y, s):
    """resync_decoder.py::atk_pad_delay: zero-pad the head, truncate to keep length."""
    out = np.zeros_like(y)
    out[s:] = y[: len(y) - s]
    return out


@pytest.fixture(scope="module")
def signed0(clip0):
    return sign(clip0, 44100, SK, NIPS_MSG)


@pytest.mark.slow
@pytest.mark.parametrize("shift", [5000, 2048 * 3])
def test_resync_recovers_delay(signed0, shift):
    yd = pad_delay(signed0, shift)
    if shift % 2048:
        assert not verify(yd, 44100, PK).verified
    t0 = time.perf_counter()
    r = verify(yd, 44100, PK, resync=True)
    print(f"resync shift={shift}: {time.perf_counter() - t0:.2f}s", r)
    assert r.verified and r.path == "resync" and r.message == NIPS_MSG.encode()
    assert f"delta={shift % 2048}" in r.reason and f"f0={shift // 2048}" in r.reason


def test_resync_is_not_used_when_the_plain_path_verifies(signed0):
    r = verify(signed0, 44100, PK, resync=True)
    assert r.verified and r.path == "header"


@pytest.mark.slow
def test_resync_negative_unsigned(clip0):
    t0 = time.perf_counter()
    r = verify(clip0, 44100, PK, resync=True)
    print(f"resync unsigned: {time.perf_counter() - t0:.2f}s", r)
    assert not r.verified and r.message is None
