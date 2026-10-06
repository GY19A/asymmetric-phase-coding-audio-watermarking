"""Test 4: v1 sign -> verify on clean audio and after int16 / MP3 / OGG / head crop / tail crop."""
import numpy as np
import pytest

from apcaw import Options, WB, keygen, sign, verify
from apcaw.layout import seed_from_pk
from apcaw.payload import parse_payload
from apcaw.verify import extract_payload_candidates, read_soft

from conftest import (FIXED_SECRET, NIPS_MSG, crop_head, crop_tail,
                      mp3_128, ogg_128, pem_pair, quantize)

SK, PK = keygen(FIXED_SECRET)


@pytest.fixture(scope="module")
def signed(clips):
    return [sign(x, 44100, SK, NIPS_MSG) for x in clips]


def _check(y, **kw):
    r = verify(y, 44100, PK, **kw)
    assert r.verified, r
    assert r.message == NIPS_MSG.encode()
    return r


def test_output_shape_and_determinism(clips, signed):
    for x, y in zip(clips, signed):
        assert y.dtype == np.float64 and y.shape == x.shape
    again = sign(clips[0], 44100, SK, NIPS_MSG)
    assert again.tobytes() == signed[0].tobytes()


def test_tail_passthrough(clips, signed):
    n = (len(clips[0]) // 2048) * 2048
    for x, y in zip(clips, signed):
        assert np.array_equal(y[n:], x[n:])


def test_clean(signed):
    for y in signed:
        r = _check(y)
        assert (r.channel, r.path, r.profile, r.rs_corrected) == ("phase", "header", "wb", 0)
        assert r.candidates_tried == 1 and r.rs_passes == 1 and r.sig_checks == 1


def test_clean_profile_autodetect_vs_explicit(signed):
    assert verify(signed[0], 44100, PK, profile=WB).verified


def test_int16(signed):
    for y in signed:
        r = _check(quantize(y).astype(np.float64) / 32768.0)
        assert r.channel == "phase"


def _channel_alone(y, channel):
    """First accepted candidate of one wb channel read on its own: (path, rs_corrected) or None."""
    ps, ms = read_soft(y, seed_from_pk(PK), WB, Options())
    soft, rmax = (ps, WB.rp) if channel == "phase" else (ms, WB.rm)
    for path, _, cand in extract_payload_candidates(soft, rmax):
        res = parse_payload(cand, PK)
        if res.ok:
            assert res.message == NIPS_MSG.encode()
            return path, res.rs_corrected
    return None


# verify() returns the first channel that verifies (phase before magnitude), and phase often
# still survives a 128 kb/s codec (clip 0000 after MP3: 15 corrected symbols, the RS limit);
# the robustness claim is that the magnitude channel carries the payload on its own
@pytest.mark.slow
@pytest.mark.parametrize("codec", ["mp3", "ogg"])
def test_codec_128k(signed, codec):
    for y in signed:
        a = mp3_128(y) if codec == "mp3" else ogg_128(y)
        r = _check(a)
        ph, mg = _channel_alone(a, "phase"), _channel_alone(a, "magnitude")
        print(f"{codec}: verify {r.channel}/{r.path} rs_corrected={r.rs_corrected}; "
              f"phase alone {ph}; magnitude alone {mg}")
        assert mg is not None
        assert r.channel == ("phase" if ph is not None else "magnitude")


def test_tail_crop_50(signed):
    for y in signed:
        _check(crop_tail(y, 50))


