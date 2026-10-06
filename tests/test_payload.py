"""Test 2: payload bytes equal legacy CryptoManager; parse round trip; RS capacity boundary."""
import numpy as np
import pytest

from apcaw import keygen
from apcaw.payload import build_payload, parse_payload

from conftest import FIXED_SECRET, NIPS_MSG, OTHER_SECRET, pem_pair

MESSAGES = ["", "A", NIPS_MSG, "x" * 159, "héllo wörld ✓"]


def test_keys_accept_pem_and_raw():
    priv_pem, pub_pem = pem_pair(FIXED_SECRET)
    sk, pk = keygen(FIXED_SECRET)
    assert build_payload(NIPS_MSG, priv_pem) == build_payload(NIPS_MSG, sk)
    assert parse_payload(build_payload(NIPS_MSG, sk), pub_pem).ok


def test_message_too_long():
    sk, _ = keygen(FIXED_SECRET)
    build_payload("y" * 159, sk)
    with pytest.raises(ValueError):
        build_payload("y" * 160, sk)


def test_parse_round_trip_and_counters():
    sk, pk = keygen(FIXED_SECRET)
    p = build_payload(NIPS_MSG, sk)
    r = parse_payload(p, pk)
    assert r.ok and r.rs_ok and r.sig_checked and r.rs_corrected == 0
    assert r.message == NIPS_MSG.encode()


@pytest.mark.parametrize("n_err,ok", [(1, True), (15, True), (16, False), (30, False)])
def test_tamper_rs_boundary(n_err, ok):
    sk, pk = keygen(FIXED_SECRET)
    p = bytearray(build_payload(NIPS_MSG, sk))
    rng = np.random.RandomState(n_err)
    for pos in rng.choice(len(p), n_err, replace=False):
        p[pos] ^= 1 + int(rng.randint(255))
    r = parse_payload(bytes(p), pk)
    assert r.ok is ok
    if ok:
        assert r.rs_corrected == n_err and r.message == NIPS_MSG.encode()
    else:
        assert r.message is None


def test_wrong_key_rs_ok_but_signature_fails():
    sk, _ = keygen(FIXED_SECRET)
    _, pk_other = keygen(OTHER_SECRET)
    r = parse_payload(build_payload(NIPS_MSG, sk), pk_other)
    assert r.rs_ok and r.sig_checked and not r.ok


def test_signature_tamper_is_rejected_after_rs():
    """Flip the length field inside a *valid* RS codeword: RS passes, length check fails."""
    from reedsolo import RSCodec
    sk, pk = keygen(FIXED_SECRET)
    c = bytearray(build_payload(NIPS_MSG, sk)[:-30])
    c[1] ^= 0x01
    r = parse_payload(bytes(RSCodec(30).encode(bytes(c))), pk)
    assert r.rs_ok and not r.sig_checked and not r.ok
