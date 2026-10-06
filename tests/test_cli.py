"""Test 9: CLI exit codes, table driven. Each case runs the installed console script (or
`python -m apcaw`) as a subprocess with stdout/stderr redirected to files (no pipes)."""
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest

from conftest import CLIP_NAMES, DATA_DIR, NIPS_MSG, ROOT, write_pcm16

APCAW = ROOT / ".venv" / "bin" / "apcaw"
RUNNERS = {"script": [str(APCAW)], "module": [sys.executable, "-m", "apcaw"]}


def run(args, tmp, runner="script"):
    out, err = tmp / "stdout.txt", tmp / "stderr.txt"
    with open(out, "wb") as fo, open(err, "wb") as fe:
        p = subprocess.run(RUNNERS[runner] + [str(a) for a in args], stdout=fo, stderr=fe,
                           stdin=subprocess.DEVNULL, timeout=600)
    return p.returncode, out.read_text(), err.read_text()


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    """Keys, a clean clip, a signed clip, a too-short clip, a legacy-signed clip."""
    d = tmp_path_factory.mktemp("cli")
    shutil.copy(DATA_DIR / CLIP_NAMES[0], d / "clip.wav")
    write_pcm16(d / "short.wav", np.random.RandomState(0).randn(2048 * 12) * 0.05)
    (d / "garbage.wav").write_bytes(b"not a wav file at all")
    (d / "badkey.bin").write_bytes(b"\x01\x02\x03")
    steps = [
        ["keygen", "--secret-out", d / "sk.bin", "--public-out", d / "pk.bin"],
        ["keygen", "--secret-out", d / "sk2.bin", "--public-out", d / "pk2.bin"],
        ["keygen", "--secret-out", d / "sk.pem", "--public-out", d / "pk.pem", "--pem"],
        ["sign", "-i", d / "clip.wav", "-o", d / "signed.wav", "-k", d / "sk.bin", "-m", NIPS_MSG],
        ["sign", "-i", d / "clip.wav", "-o", d / "signed_nb.wav", "-k", d / "sk.bin", "-m", "nb",
         "--profile", "nb"],
        ["sign", "-i", d / "clip.wav", "-o", d / "signed_pem.wav", "-k", d / "sk.pem", "-m", "pem"],
        ["sign", "-i", d / "clip.wav", "-o", d / "signed_legacy.wav", "-k", d / "sk.bin",
         "-m", NIPS_MSG, "--legacy"],
    ]
    for s in steps:
        rc, o, e = run(s, d)
        assert rc == 0, (s, o, e)
    return d


CASES = [
    # (id, args, expected exit code)
    ("verify-signed", ["verify", "-i", "signed.wav", "-p", "pk.bin"], 0),
    ("verify-signed-json", ["verify", "-i", "signed.wav", "-p", "pk.bin", "--json"], 0),
    ("verify-pem-key", ["verify", "-i", "signed_pem.wav", "-p", "pk.pem"], 0),
    ("verify-pem-other-pair", ["verify", "-i", "signed_pem.wav", "-p", "pk.bin"], 1),
    ("verify-nb", ["verify", "-i", "signed_nb.wav", "-p", "pk.bin", "--profile", "nb"], 0),
    ("verify-nb-auto", ["verify", "-i", "signed_nb.wav", "-p", "pk.bin"], 0),
    ("verify-legacy", ["verify", "-i", "signed_legacy.wav", "-p", "pk.bin", "--legacy"], 0),
    ("verify-legacy-as-v1", ["verify", "-i", "signed_legacy.wav", "-p", "pk.bin"], 1),
    ("verify-unsigned", ["verify", "-i", "clip.wav", "-p", "pk.bin"], 1),
    ("verify-unsigned-json", ["verify", "-i", "clip.wav", "-p", "pk.bin", "--json"], 1),
    ("verify-wrong-key", ["verify", "-i", "signed.wav", "-p", "pk2.bin"], 1),
    ("verify-resync-clean", ["verify", "-i", "signed.wav", "-p", "pk.bin", "--resync"], 0),
    ("verify-missing-input", ["verify", "-i", "nope.wav", "-p", "pk.bin"], 3),
    ("verify-garbage-input", ["verify", "-i", "garbage.wav", "-p", "pk.bin"], 3),
    ("verify-missing-key", ["verify", "-i", "signed.wav", "-p", "nope.bin"], 3),
    ("verify-malformed-key", ["verify", "-i", "signed.wav", "-p", "badkey.bin"], 2),
    ("verify-no-key-arg", ["verify", "-i", "signed.wav"], 2),
    ("verify-bad-profile", ["verify", "-i", "signed.wav", "-p", "pk.bin", "--profile", "xx"], 2),
    ("sign-too-long", ["sign", "-i", "clip.wav", "-o", "o1.wav", "-k", "sk.bin", "-m", "z" * 160], 2),
    ("sign-too-short", ["sign", "-i", "short.wav", "-o", "o2.wav", "-k", "sk.bin", "-m", "hi"], 1),
    ("sign-missing-input", ["sign", "-i", "nope.wav", "-o", "o3.wav", "-k", "sk.bin", "-m", "hi"], 3),
    ("sign-missing-key", ["sign", "-i", "clip.wav", "-o", "o4.wav", "-k", "nope.bin", "-m", "hi"], 3),
    ("sign-public-as-secret", ["sign", "-i", "clip.wav", "-o", "o5.wav", "-k", "pk.pem", "-m", "hi"], 2),
    ("inspect", ["inspect", "-i", "signed.wav", "-p", "pk.bin", "--json"], 0),
    ("layout", ["layout", "-p", "pk.bin", "--json"], 0),
    ("selftest", ["selftest"], 0),
    ("no-args", [], 2),
    ("unknown-command", ["frobnicate"], 2),
]


# every case through the console script; `python -m apcaw` on a subset
MODULE_CASES = ("verify-signed", "verify-unsigned", "selftest", "no-args")
RUNS = [(r, *c) for c in CASES for r in ("script", "module") if r == "script" or c[0] in MODULE_CASES]


@pytest.mark.parametrize("runner,name,args,code", RUNS, ids=[f"{c[1]}-{c[0]}" for c in RUNS])
def test_exit_codes(ws, runner, name, args, code):
    args = [ws / a if (ws / a).exists() or a.endswith((".wav", ".bin", ".pem")) else a
            for a in args]
    rc, out, err = run(args, ws, runner)
    assert rc == code, (name, rc, out, err)


def test_failed_sign_leaves_no_output(ws):
    for name, args, code in CASES:
        if name.startswith("sign-"):
            rc, _, _ = run([ws / a if a.endswith((".wav", ".bin", ".pem")) else a for a in args], ws)
            assert rc == code != 0
            assert not (ws / args[args.index("-o") + 1]).exists()
    assert not [p for p in ws.iterdir() if ".tmp" in p.name]


def test_verify_json_fields(ws):
    rc, out, _ = run(["verify", "-i", ws / "signed.wav", "-p", ws / "pk.bin", "--json"], ws)
    j = json.loads(out)
    assert rc == 0 and j["verified"] is True
    assert j["message"] == NIPS_MSG and j["message_hex"] == NIPS_MSG.encode().hex()
    assert j["channel"] == "phase" and j["path"] == "header" and j["profile"] == "wb"
    for k in ("rs_corrected", "candidates_tried", "rs_passes", "sig_checks", "reason"):
        assert k in j
    rc, out, _ = run(["verify", "-i", ws / "clip.wav", "-p", ws / "pk.bin", "--json"], ws)
    j = json.loads(out)
    assert rc == 1 and j["verified"] is False and j["message"] is None and j["reason"]


def test_keys_on_disk(ws):
    assert (ws / "sk.bin").stat().st_size == 32 and (ws / "pk.bin").stat().st_size == 32
    assert (ws / "sk.bin").stat().st_mode & 0o077 == 0
    assert (ws / "sk.pem").read_bytes().startswith(b"-----BEGIN PRIVATE KEY-----")
    assert (ws / "pk.pem").read_bytes().startswith(b"-----BEGIN PUBLIC KEY-----")
    # the secret never appears on stdout / stderr
    rc, out, err = run(["keygen", "--secret-out", ws / "sk3.bin", "--public-out", ws / "pk3.bin"], ws)
    sk_hex = (ws / "sk3.bin").read_bytes().hex()
    assert rc == 0 and sk_hex not in out and sk_hex not in err


def test_signed_wav_is_pcm16_and_deterministic(ws):
    import wave
    with wave.open(str(ws / "signed.wav"), "rb") as w:
        assert (w.getsampwidth(), w.getnchannels(), w.getframerate()) == (2, 1, 44100)
    rc, _, _ = run(["sign", "-i", ws / "clip.wav", "-o", ws / "signed_again.wav", "-k", ws / "sk.bin",
                    "-m", NIPS_MSG], ws)
    assert rc == 0
    h = [hashlib.sha256((ws / n).read_bytes()).hexdigest() for n in ("signed.wav", "signed_again.wav")]
    assert h[0] == h[1]


def test_cli_output_equals_library(ws):
    """The CLI's WAV is exactly quantize(sign(...)) of the library call."""
    from apcaw import sign
    from apcaw.io import quantize_pcm16, read_audio
    from apcaw.payload import load_secret_key
    x, sr = read_audio(ws / "clip.wav")
    y = sign(x, sr, load_secret_key((ws / "sk.bin").read_bytes()), NIPS_MSG)
    ref = quantize_pcm16(y)
    got, _ = read_audio(ws / "signed.wav")
    assert np.array_equal(np.round(got * 32768).astype(np.int16), ref)


def test_inspect_and_layout_json(ws):
    rc, out, _ = run(["inspect", "-i", ws / "signed.wav", "-p", ws / "pk.bin", "--json"], ws)
    j = json.loads(out)
    assert rc == 0
    wb = j["profiles"]["wb"]
    assert wb["groups"] == 26 and wb["phase"]["capacity"] == 6240 and wb["magnitude"]["capacity"] == 3120
    assert wb["phase"]["header_length"] == 8 * (len(NIPS_MSG) + 96)
    assert wb["phase"]["mean_abs_soft"] > 0.9
    rc, out, _ = run(["layout", "-p", ws / "pk.bin", "--json"], ws)
    j = json.loads(out)
    assert rc == 0 and len(j["wb"]["phase_bins"]) == 240 and len(j["nb"]["mag_pairs"]) == 76
    assert j["seed"] == int.from_bytes(hashlib.sha256((ws / "pk.bin").read_bytes()).digest()[:8],
                                       "big") % 2**32
