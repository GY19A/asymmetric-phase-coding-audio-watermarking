"""Keys and payload (format §4): C = be16(len M) || M || Ed25519(M); payload = RS(30) encode of C."""
from __future__ import annotations

import os
from typing import NamedTuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from reedsolo import ReedSolomonError, RSCodec

from .params import MAX_MSG_LEN, RS_NSYM, SIG_LEN

_RSC = RSCodec(RS_NSYM)


# ----------------------------------------------------------------------------------------------
# Keys: raw 32-byte Ed25519 seed / public key; files may also hold 64 hex digits or PEM
# ----------------------------------------------------------------------------------------------
def keygen(seed: bytes | None = None) -> tuple[bytes, bytes]:
    """(secret seed 32 B, public key 32 B). `seed` fixes the secret (tests, vectors)."""
    if seed is None:
        seed = os.urandom(32)
    sk = load_secret_key(seed)
    return sk, public_key(sk)


def public_key(sk) -> bytes:
    return _sk_obj(load_secret_key(sk)).public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def _maybe_hex(data: bytes) -> bytes | None:
    s = data.strip()
    if len(s) == 64:
        try:
            return bytes.fromhex(s.decode("ascii"))
        except (UnicodeDecodeError, ValueError):
            return None
    return None


def load_secret_key(data) -> bytes:
    """Raw 32-byte seed from raw bytes, 64 hex digits, PKCS8 PEM, or an Ed25519PrivateKey."""
    if isinstance(data, ed25519.Ed25519PrivateKey):
        return data.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                  serialization.NoEncryption())
    if isinstance(data, str):
        data = data.encode("ascii")
    data = bytes(data)
    if len(data) == 32:
        return data
    if data.lstrip().startswith(b"-----BEGIN"):
        try:
            k = serialization.load_pem_private_key(data, password=None)
        except (ValueError, TypeError) as e:
            raise ValueError(f"not an unencrypted Ed25519 private key PEM: {e}") from None
        if not isinstance(k, ed25519.Ed25519PrivateKey):
            raise ValueError("PEM private key is not Ed25519")
        return load_secret_key(k)
    h = _maybe_hex(data)
    if h is not None:
        return h
    raise ValueError(f"malformed secret key ({len(data)} bytes; expected raw 32 B, 64 hex, or PEM)")


def load_public_key(data) -> bytes:
    """Raw 32-byte public key from raw bytes, 64 hex digits, SPKI PEM, or an Ed25519PublicKey."""
    if isinstance(data, ed25519.Ed25519PublicKey):
        return data.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    if isinstance(data, str):
        data = data.encode("ascii")
    data = bytes(data)
    if len(data) == 32:
        return data
    if data.lstrip().startswith(b"-----BEGIN"):
        try:
            k = serialization.load_pem_public_key(data)
        except (ValueError, TypeError) as e:
            raise ValueError(f"not an Ed25519 public key PEM: {e}") from None
        if not isinstance(k, ed25519.Ed25519PublicKey):
            raise ValueError("PEM public key is not Ed25519")
        return load_public_key(k)
    h = _maybe_hex(data)
    if h is not None:
        return h
    raise ValueError(f"malformed public key ({len(data)} bytes; expected raw 32 B, 64 hex, or PEM)")


def _sk_obj(sk32: bytes) -> ed25519.Ed25519PrivateKey:
    return ed25519.Ed25519PrivateKey.from_private_bytes(sk32)


def secret_pem(sk) -> bytes:
    return _sk_obj(load_secret_key(sk)).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())


def public_pem(pk) -> bytes:
    return ed25519.Ed25519PublicKey.from_public_bytes(load_public_key(pk)).public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)


# ----------------------------------------------------------------------------------------------
# Payload
# ----------------------------------------------------------------------------------------------
def message_bytes(message: bytes | str) -> bytes:
    m = message.encode("utf-8") if isinstance(message, str) else bytes(message)
    if len(m) > MAX_MSG_LEN:
        raise ValueError(f"message too long: {len(m)} bytes > {MAX_MSG_LEN}")
    return m


def build_payload(message: bytes | str, sk) -> bytes:
    m = message_bytes(message)
    sig = _sk_obj(load_secret_key(sk)).sign(m)
    return bytes(_RSC.encode(len(m).to_bytes(2, "big") + m + sig))


class ParseResult(NamedTuple):
    ok: bool
    message: bytes | None
    rs_ok: bool             # Reed-Solomon decoding succeeded
    sig_checked: bool       # the length field was consistent, so Ed25519 was evaluated
    rs_corrected: int | None
    reason: str


def rs_decode(data: bytes):
    """(decoded bytes, number of corrected symbols) or None. reedsolo semantics, including its
    chunking for inputs longer than 255 bytes (only reachable from the legacy header path)."""
    try:
        msg, _, errata = _RSC.decode(bytes(data))
    except ReedSolomonError:
        return None
    return bytes(msg), len(errata)


def parse_payload(data: bytes, pk) -> ParseResult:
    pk32 = load_public_key(pk)
    dec = rs_decode(data)
    if dec is None:
        return ParseResult(False, None, False, False, None, "RS decoding failed")
    c, ncorr = dec
    if len(c) < 2 or len(c) != 2 + int.from_bytes(c[:2], "big") + SIG_LEN:
        return ParseResult(False, None, True, False, ncorr, "length field inconsistent")
    mlen = int.from_bytes(c[:2], "big")
    m, sig = c[2:2 + mlen], c[2 + mlen:]
    try:
        ed25519.Ed25519PublicKey.from_public_bytes(pk32).verify(sig, m)
    except InvalidSignature:
        return ParseResult(False, None, True, True, ncorr, "signature invalid")
    return ParseResult(True, m, True, True, ncorr, "ok")
