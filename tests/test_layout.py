"""Test 1: key-derived layout equals the legacy HybridCoder patterns; MT19937 / shuffle reference."""
import hashlib

import numpy as np
import pytest

from apcaw import NB, WB, keygen
from apcaw.layout import (mag_pairs, mt19937_outputs, phase_bins, py_shuffle,
                          seed_from_pk)

from conftest import FIXED_SECRET

_, PK_FIXED = keygen(FIXED_SECRET)
SEEDS = [0, 1, 42, 2**32 - 1, seed_from_pk(PK_FIXED)]


def test_seed_from_pk_matches_spec_formula():
    expect = int.from_bytes(hashlib.sha256(PK_FIXED).digest()[:8], "big") % 2**32
    assert seed_from_pk(PK_FIXED) == expect


@pytest.mark.parametrize("seed", SEEDS + [(s ^ 0xDEADBEEF) & 0xFFFFFFFF for s in SEEDS])
def test_mt19937_pure_python_matches_numpy(seed):
    ref = np.random.RandomState(seed).randint(0, 2**32, size=16, dtype=np.uint32)
    assert mt19937_outputs(seed, 16) == [int(v) for v in ref]


def test_mt19937_known_first_outputs():
    # values quoted in the Rust/JS task briefs
    assert mt19937_outputs(42, 1) == [1608637542]
    assert mt19937_outputs(0, 1) == [2357136044]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n", [2, 3, 76, 120, 240])
def test_pure_python_shuffle_matches_randomstate(seed, n):
    a = np.arange(n)
    np.random.RandomState(seed).shuffle(a)
    assert py_shuffle(seed, list(range(n))) == a.tolist()


def test_layout_depends_on_key():
    _, pk2 = keygen(bytes(range(1, 33)))
    s1, s2 = seed_from_pk(PK_FIXED), seed_from_pk(pk2)
    assert s1 != s2
    assert not np.array_equal(phase_bins(s1, WB), phase_bins(s2, WB))
