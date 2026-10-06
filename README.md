# Asymmetric Phase Coding Audio Watermarking

**Sign audio with a secret key. Verify it with the public key alone.**

This is the Python reference implementation of APC, the audio watermark of the NeurIPS 2026 paper
*Asymmetric Phase Coding Audio Watermarking*. An Ed25519-signed message is embedded in the STFT
phase and magnitude of 44.1 kHz audio. A verifier that holds only the 32-byte public key recovers
the message and checks the signature. It needs no original, no model weights, no registry and no
shared secret. The method is training-free.

- Paper: [NeurIPS 2026](https://neurips.cc/virtual/2026/loc/atlanta/poster/149813),
  [arXiv:2605.07241](https://arxiv.org/abs/2605.07241)
- Live demo, running entirely in your browser:
  <https://gy19a.github.io/asymmetric-phase-coding-audio-watermarking-js/>

## Three implementations, one format

| Repository | Language | What it is |
| --- | --- | --- |
| **this repository** | Python | Reference implementation, CLI, and the conformance vectors |
| [asymmetric-phase-coding-audio-watermarking-rs](https://github.com/GY19A/asymmetric-phase-coding-audio-watermarking-rs) | Rust | Library, `apcaw` CLI and C ABI; a byte-exact port |
| [asymmetric-phase-coding-audio-watermarking-js](https://github.com/GY19A/asymmetric-phase-coding-audio-watermarking-js) | JavaScript | Library for Node and browsers, and the demo site |

All three read and write the same format, `apcaw-v1`, and pass the same conformance vectors in
[`vectors/`](vectors/). Files signed by one verify in the others, and the Rust port writes
byte-identical signed WAV files.

## Install

```sh
python -m venv .venv
.venv/bin/pip install -e ".[cli,test]"
```

Runtime dependencies: numpy, cryptography and reedsolo.

- Without extras, input must be 16-bit PCM WAV (stdlib `wave`).
- `soundfile` (the `[cli]` extra) reads any libsndfile format.
- Multichannel input is downmixed to mono, with a notice.
- Input that is not 44.1 kHz is resampled with `scipy.signal.resample_poly` if scipy is
  installed, with a notice. Otherwise it is rejected (exit 3).
- Output is always 44.1 kHz mono 16-bit PCM WAV.

## Library

```python
from apcaw import keygen, sign, verify
from apcaw.io import read_audio

x, sr = read_audio("in.wav")
sk, pk = keygen()                          # 32-byte Ed25519 seed, 32-byte public key
y = sign(x, sr, sk, "hello")               # float64 in, float64 out, same length
r = verify(y, sr, pk)                      # profile auto (wb, then nb)
print(r.verified, r.message.decode(), r.channel, r.rs_corrected)
verify(y_shifted, sr, pk, resync=True)     # optional sample / frame offset search
```

`Options()` is the v1 configuration. `Options.legacy()` reproduces the original configuration used
for the paper's legacy numbers: seed 42, 8 phase frames, header-only verifier, zeroed tail, no
erasures.

## Command line

```sh
apcaw keygen --secret-out sk.bin --public-out pk.bin [--pem]
apcaw sign   -i in.wav -o out.wav -k sk.bin -m "message" [--profile wb|nb] [--legacy]
apcaw verify -i out.wav -p pk.bin [--profile wb|nb] [--resync] [--json] [--legacy]
apcaw inspect -i out.wav -p pk.bin [--json]
apcaw layout  -p pk.bin | --seed N [--json]
apcaw selftest
```

`sign` is closed loop: it writes 16-bit PCM, reads the file back and exits 0 only if the file
verifies. Otherwise it writes nothing. Exit codes for `verify`: 0 verified, 1 not verified,
2 usage error (including a malformed key), 3 I/O error. `selftest` checks the build against a
bundled copy of the layout, MT19937, payload and Reed-Solomon vectors, plus one synthetic signing.

## How it works

- **Phase channel.** Each bit sets the phase of one key-selected STFT cell per group of eight
  frames to +π/2 (bit 1) or −π/2 (bit 0). The verifier reads the sign of sin φ.
- **Magnitude channel.** Each bit moves the log-level difference of a key-selected bin pair to an
  odd (1) or even (0) multiple of one nat. The verifier reads its parity.
- **Payload.** Length, message and a 64-byte Ed25519 signature, protected by 30 Reed-Solomon
  parity bytes (up to 15 byte errors repaired).
- **Layout.** The bins and pairs come from a shuffle seeded by the SHA-256 of the public key, so
  the verifier rebuilds the layout from the key and nothing is stored or sent.
- **Acceptance.** A candidate is accepted only if its Ed25519 signature verifies. A false accept
  requires an Ed25519 forgery.

The full format is specified in Appendix A of the paper.

## Tests and conformance vectors

```sh
.venv/bin/python -m pytest -q             # all tests; -m "not slow" for the fast subset
.venv/bin/python tools/make_vectors.py    # regenerate vectors/ and src/apcaw/data/selftest.json
```

- The `slow` and codec tests need `ffmpeg`.
- Temporary files stay under `.tmp/`.
- `vectors/` is frozen. `make_vectors.py` is deterministic: a rerun reproduces every file byte for
  byte, given the ffmpeg build recorded in `vectors/MANIFEST.json`. The Rust and JavaScript
  repositories carry checked copies of these files.

## Citation

```bibtex
@inproceedings{yang2026apc,
  title     = {Asymmetric Phase Coding Audio Watermarking},
  author    = {Yang, Guang and Liu, Fengchen and Ghasemian, Amir and Wang, Zhong and
               Mehrabi, Ninareh and Hosseinmardi, Homa},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

BSD 2-Clause, see [`LICENSE`](LICENSE). Copyright (c) 2026, Guang Yang (guangyang19@ucla.edu).

The audio in `tests/media/` and `vectors/` is derived from LibriSpeech (Panayotov et al.,
ICASSP 2015), licensed CC BY 4.0; see `tests/media/ATTRIBUTION.txt` and `vectors/README.md`.
