"""apcaw command line. Exit codes: 0 success / verified, 1 not verified or signing failure,
2 usage error (bad arguments, malformed key, message too long), 3 I/O error."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import __version__
from .embed import sign
from .io import AudioIOError, quantize_pcm16, read_audio, to_44100, write_pcm16
from .layout import capacity, mag_pairs, n_frames, phase_bins, replicas, seed_from_pk
from .params import G, NB, PROFILES, SR, WB, Options, get_profile
from .payload import (keygen, load_public_key, load_secret_key, message_bytes, public_key,
                      public_pem, secret_pem)
from .verify import MIN_PAYLOAD_BITS, header_length, read_soft, verify

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_IO = 0, 1, 2, 3


class CliError(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


def _err(msg: str) -> None:
    print(f"apcaw: {msg}", file=sys.stderr)


def _read_key_file(path, loader, what):
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        raise CliError(EXIT_IO, f"cannot read {what} {path}: {e.strerror or e}") from None
    try:
        return loader(data)
    except ValueError as e:
        raise CliError(EXIT_USAGE, f"{path}: {e}") from None


def _load_input(path, notices):
    try:
        x, sr = read_audio(path, notices)
        return to_44100(x, sr, notices)
    except AudioIOError as e:
        raise CliError(EXIT_IO, str(e)) from None


def _show_notices(notices):
    for n in notices:
        print(f"apcaw: note: {n}", file=sys.stderr)


def _options(args) -> Options:
    return Options.legacy() if getattr(args, "legacy", False) else Options()


# ----------------------------------------------------------------------------------------------
def cmd_keygen(args) -> int:
    sk, pk = keygen()
    if args.pem:
        sk_data, pk_data = secret_pem(sk), public_pem(pk)
    else:
        sk_data, pk_data = sk, pk
    for p in (args.secret_out, args.public_out):
        if os.path.exists(p) and not args.force:
            raise CliError(EXIT_USAGE, f"{p} exists (use --force to overwrite)")
    try:
        fd = os.open(args.secret_out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            os.fchmod(f.fileno(), 0o600)
            f.write(sk_data)
        with open(args.public_out, "wb") as f:
            f.write(pk_data)
    except OSError as e:
        raise CliError(EXIT_IO, f"cannot write key: {e}") from None
    print(f"public key {pk.hex()}")
    print(f"layout seed {seed_from_pk(pk)}")
    print(f"wrote {args.secret_out} (secret, mode 0600) and {args.public_out}")
    return EXIT_OK


def cmd_sign(args) -> int:
    notices: list[str] = []
    try:
        msg = message_bytes(args.message)
    except ValueError as e:
        raise CliError(EXIT_USAGE, str(e)) from None
    profile = get_profile(args.profile)
    options = _options(args)
    x = _load_input(args.input, notices)
    sk = _read_key_file(args.key, load_secret_key, "secret key")
    _show_notices(notices)
    try:
        y = sign(x, SR, sk, msg, profile, options)
    except ValueError as e:
        _err(f"signing failed: {e}")
        return EXIT_FAIL
    q = quantize_pcm16(y)
    n_clip = int(np.count_nonzero((y < -1.0) | (y > 32767 / 32768)))
    if n_clip:
        _err(f"note: {n_clip} samples clipped to [-1, 1)")

    out = os.path.abspath(args.output)
    tmp = os.path.join(os.path.dirname(out), f".{os.path.basename(out)}.tmp{os.getpid()}")
    try:
        write_pcm16(tmp, q, SR)
        # closed loop (format §9): the written file must verify
        back, _ = read_audio(tmp)
        r = verify(back, SR, public_key(sk), profile, options)
        if not (r.verified and r.message == msg):
            _err(f"signing failed: the written file does not verify ({r.reason}); "
                 f"nothing written to {args.output}")
            return EXIT_FAIL
        os.replace(tmp, out)
    except AudioIOError as e:
        raise CliError(EXIT_IO, str(e)) from None
    except OSError as e:
        raise CliError(EXIT_IO, f"cannot write {args.output}: {e}") from None
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    print(f"signed {args.output}: {len(msg)}-byte message, profile {profile.name}, "
          f"closed-loop verify ok (channel {r.channel}, path {r.path})")
    return EXIT_OK


def _result_json(r, path) -> dict:
    return {
        "file": os.fspath(path),
        "verified": r.verified,
        "message": r.message.decode("utf-8", errors="replace") if r.message is not None else None,
        "message_hex": r.message.hex() if r.message is not None else None,
        "channel": r.channel, "profile": r.profile, "path": r.path,
        "rs_corrected": r.rs_corrected, "payload_bits": r.payload_bits,
        "candidates_tried": r.candidates_tried, "rs_passes": r.rs_passes,
        "sig_checks": r.sig_checks, "reason": r.reason,
    }


def cmd_verify(args) -> int:
    notices: list[str] = []
    x = _load_input(args.input, notices)
    pk = _read_key_file(args.public_key, load_public_key, "public key")
    _show_notices(notices)
    r = verify(x, SR, pk, args.profile, _options(args), resync=args.resync)
    if args.json:
        print(json.dumps(_result_json(r, args.input), indent=2))
    elif r.verified:
        print(f"VERIFIED  message={r.message.decode('utf-8', errors='replace')!r}  "
              f"channel={r.channel} profile={r.profile} path={r.path} rs_corrected={r.rs_corrected}")
    else:
        print(f"NOT VERIFIED  {r.reason}")
    return EXIT_OK if r.verified else EXIT_FAIL


def cmd_inspect(args) -> int:
    notices: list[str] = []
    x = _load_input(args.input, notices)
    pk = _read_key_file(args.public_key, load_public_key, "public key")
    _show_notices(notices)
    options = _options(args)
    seed = options.seed if options.seed is not None else seed_from_pk(pk)
    rep = {"file": os.fspath(args.input), "samples": len(x), "frames": n_frames(len(x)),
           "seed": seed, "profiles": {}}
    for name, prof in PROFILES.items():
        cap = capacity(len(x), prof)
        ps, ms = read_soft(x, seed, prof, options)
        ent = {"groups": cap.groups, "frames": cap.frames}
        for ch, soft, rmax in (("phase", ps, prof.rp), ("magnitude", ms, prof.rm)):
            ell = header_length(soft)
            # the stream the header announces, if it is a plausible payload length
            n = len(soft)
            if ell is not None and ell % 8 == 0 and MIN_PAYLOAD_BITS <= ell <= len(soft) - 96:
                n = 96 + replicas(len(soft), ell, rmax) * ell
            ent[ch] = {"capacity": len(soft), "header_length": ell, "stream_bits": n,
                       "mean_abs_soft": float(np.mean(np.abs(soft[:n]))) if n else None,
                       "mean_abs_soft_capacity": float(np.mean(np.abs(soft))) if len(soft) else None,
                       "erased": int(np.count_nonzero(soft == 0.0))}
        rep["profiles"][name] = ent
    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        print(f"{rep['file']}: {rep['samples']} samples, {rep['frames']} frames, seed {seed}")
        for name, ent in rep["profiles"].items():
            for ch in ("phase", "magnitude"):
                e = ent[ch]
                print(f"  {name} {ch:9s} capacity {e['capacity']:5d}  header_length "
                      f"{e['header_length']}  mean|soft| over {e['stream_bits']} bits "
                      f"{e['mean_abs_soft']:.4f}  erased {e['erased']}")
    return EXIT_OK


def cmd_layout(args) -> int:
    if args.public_key is not None:
        seed = seed_from_pk(_read_key_file(args.public_key, load_public_key, "public key"))
    else:
        seed = args.seed
        if not 0 <= seed < 2**32:
            raise CliError(EXIT_USAGE, "seed must be in [0, 2**32)")
    rep = {"seed": seed}
    for name, prof in PROFILES.items():
        rep[name] = {"phase_bins": phase_bins(seed, prof).tolist(),
                     "mag_pairs": mag_pairs(seed, prof).tolist()}
    if args.json:
        print(json.dumps(rep))
    else:
        print(f"seed {seed}")
        for name in PROFILES:
            print(f"  {name} Kp[:8] {rep[name]['phase_bins'][:8]}  pairs[:4] {rep[name]['mag_pairs'][:4]}")
    return EXIT_OK


def cmd_selftest(args) -> int:
    from . import selftest
    try:
        results = selftest.run()
    except FileNotFoundError as e:
        raise CliError(EXIT_IO, f"selftest data missing: {e}") from None
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    ok = all(r[1] for r in results)
    print("selftest " + ("passed" if ok else "FAILED"))
    return EXIT_OK if ok else EXIT_FAIL


# ----------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="apcaw", description="APC audio watermarking (apcaw-v1).")
    ap.add_argument("--version", action="version", version=f"apcaw {__version__}")
    sub = ap.add_subparsers(dest="command", metavar="command")
    prof = dict(choices=sorted(PROFILES), type=str.lower)

    p = sub.add_parser("keygen", help="generate an Ed25519 key pair")
    p.add_argument("--secret-out", required=True)
    p.add_argument("--public-out", required=True)
    p.add_argument("--pem", action="store_true", help="PKCS#8 / SPKI PEM instead of raw 32 bytes")
    p.add_argument("--force", action="store_true", help="overwrite existing key files")
    p.set_defaults(func=cmd_keygen)

    p = sub.add_parser("sign", help="embed a signed message (closed loop, 16-bit PCM WAV out)")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("-k", "--key", required=True, help="secret key (raw 32 B, hex, or PEM)")
    p.add_argument("-m", "--message", required=True, help="UTF-8 message, at most 159 bytes")
    p.add_argument("--profile", default="wb", **prof)
    p.add_argument("--legacy", action="store_true", help="legacy configuration (seed 42)")
    p.set_defaults(func=cmd_sign)

    p = sub.add_parser("verify", help="verify a file against a public key")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-p", "--public-key", required=True)
    p.add_argument("--profile", default=None, **prof)
    p.add_argument("--resync", action="store_true", help="also search sample / frame offsets")
    p.add_argument("--json", action="store_true")
    p.add_argument("--legacy", action="store_true", help="legacy verifier (header only, seed 42)")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("inspect", help="capacity, header lengths and soft-value statistics")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-p", "--public-key", required=True)
    p.add_argument("--json", action="store_true")
    p.add_argument("--legacy", action="store_true")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("layout", help="print the key-derived layout")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("-p", "--public-key")
    g.add_argument("--seed", type=int)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_layout)

    p = sub.add_parser("selftest", help="check this build against frozen reference values")
    p.set_defaults(func=cmd_selftest)
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.command is None:
        ap.print_help(sys.stderr)
        return EXIT_USAGE
    try:
        return args.func(args)
    except CliError as e:
        _err(str(e))
        return e.code
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
