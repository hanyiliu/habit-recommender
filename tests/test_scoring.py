# tests/test_scoring.py
"""Regression coverage for the K-means routine builder.

`build_routines` is the one pipeline step that runs scikit-learn's K-means.
When it executes in the same process as PyTorch (train_main.py and the predict
runner both do), the two bundled OpenMP runtimes (torch's libomp,
numpy/sklearn's libiomp5) collide and K-means hangs with "OMP: Error #179"
unless the call is thread-limited. Importing torch here reproduces that process
state, so this test fails (hangs/errors) without the threadpool_limits guard in
scoring.build_routines and passes with it.
"""
import numpy as np
import torch  # noqa: F401 — imported to load torch's OpenMP runtime alongside sklearn's

from src.scoring.scoring import build_routines


def _toy_sequences(n: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 11, size=(n, 48), dtype=np.int64)


def test_build_routines_runs_with_torch_loaded():
    seqs = _toy_sequences()
    routines, labels, scores = build_routines(
        seqs, K=4, min_cluster_size=5, random_state=42,
    )
    assert routines.ndim == 2 and routines.shape[1] == 48
    assert routines.shape[0] <= 4          # R <= K (some clusters may merge)
    assert labels.shape == (seqs.shape[0],)
    assert scores.shape == (seqs.shape[0],)
    assert np.all((routines >= 0) & (routines < 11))


def test_build_routines_reproducible():
    seqs = _toy_sequences()
    r1, l1, s1 = build_routines(seqs, K=4, min_cluster_size=5, random_state=42)
    r2, l2, s2 = build_routines(seqs, K=4, min_cluster_size=5, random_state=42)
    assert np.array_equal(r1, r2)
    assert np.array_equal(l1, l2)
    assert np.allclose(s1, s2)
