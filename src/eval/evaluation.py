"""
Evaluation metrics for the temporal habit/routine recommender (Phase 5).

All functions accept numpy arrays (or array-likes). No PyTorch dependency is
required at the metric layer; teammates can hand off scores as numpy arrays
detached from any framework.

Conventions
-----------
y_true   : shape (n_samples,)              integer class labels in [0, n_classes)
y_scores : shape (n_samples, n_classes)    higher score == more recommended
sequences: 1-D integer arrays of equal length (e.g. 48 time slots per day)
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_ranking_inputs(y_true, y_scores):
    """Coerce, shape-check, and bounds-check ranking inputs."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=np.float64)

    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1-D, got shape {y_true.shape}.")
    if y_scores.ndim != 2:
        raise ValueError(f"y_scores must be 2-D, got shape {y_scores.shape}.")
    if y_true.shape[0] != y_scores.shape[0]:
        raise ValueError(
            f"Mismatched lengths: y_true has {y_true.shape[0]} samples but "
            f"y_scores has {y_scores.shape[0]}."
        )
    if y_true.size == 0:
        raise ValueError("Empty input arrays passed to metric.")

    y_true = y_true.astype(np.int64, copy=False)
    n_classes = y_scores.shape[1]
    if (y_true < 0).any() or (y_true >= n_classes).any():
        raise ValueError(
            f"y_true contains values outside the valid range [0, {n_classes})."
        )
    return y_true, y_scores


def _topk_indices(y_scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k scores per row, sorted descending. Handles k > n_classes."""
    n_classes = y_scores.shape[1]
    k_eff = min(k, n_classes)
    # argpartition is O(n); sort only the top-k afterwards for deterministic ordering.
    part = np.argpartition(-y_scores, k_eff - 1, axis=1)[:, :k_eff]
    rows = np.arange(y_scores.shape[0])[:, None]
    order = np.argsort(-y_scores[rows, part], axis=1)
    return part[rows, order]


def _validate_sequences(*seqs, name: str = "sequences"):
    """Coerce sequences to 1-D int arrays of equal length, non-empty."""
    arrs = [np.asarray(s).astype(np.int64, copy=False) for s in seqs]
    for a in arrs:
        if a.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape {a.shape}.")
    lengths = {a.shape[0] for a in arrs}
    if len(lengths) > 1:
        raise ValueError(f"Mismatched sequence lengths in {name}: {sorted(lengths)}.")
    if not arrs or arrs[0].size == 0:
        raise ValueError(f"Empty {name} passed.")
    return arrs


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def hit_rate_at_k(y_true, y_scores, k: int = 5) -> float:
    """Fraction of samples whose true class is in the top-k recommended classes."""
    y_true, y_scores = _validate_ranking_inputs(y_true, y_scores)
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    topk = _topk_indices(y_scores, k)
    hits = (topk == y_true[:, None]).any(axis=1)
    return float(hits.mean())


def ndcg_at_k(y_true, y_scores, k: int = 5) -> float:
    """
    NDCG@k for the single-relevant-item case (one true label per sample).

    With a single relevant item, IDCG = 1, so NDCG = 1 / log2(rank + 1) if the
    true class appears anywhere in the top-k, else 0. Averaged over samples.
    """
    y_true, y_scores = _validate_ranking_inputs(y_true, y_scores)
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    topk = _topk_indices(y_scores, k)
    matches = topk == y_true[:, None]
    has_match = matches.any(axis=1)
    positions = np.argmax(matches, axis=1)  # 0 when no match, but masked below
    dcg = np.where(has_match, 1.0 / np.log2(positions + 2.0), 0.0)
    return float(dcg.mean())


def next_activity_accuracy(y_true, y_scores) -> float:
    """Top-1 accuracy of the predicted next activity."""
    y_true, y_scores = _validate_ranking_inputs(y_true, y_scores)
    preds = np.argmax(y_scores, axis=1)
    return float((preds == y_true).mean())


def evaluate_ranking(
    y_true,
    y_scores,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict:
    """Compute hit_rate@k and ndcg@k for several k, plus top-1 accuracy."""
    y_true, y_scores = _validate_ranking_inputs(y_true, y_scores)
    out: dict = {"accuracy": next_activity_accuracy(y_true, y_scores)}
    for k in ks:
        out[f"hit_rate@{k}"] = hit_rate_at_k(y_true, y_scores, k=k)
        out[f"ndcg@{k}"] = ndcg_at_k(y_true, y_scores, k=k)
    return out


# ---------------------------------------------------------------------------
# Sequence metrics
# ---------------------------------------------------------------------------

def sequence_match_score(y_true_sequence, y_pred_sequence) -> float:
    """Position-wise agreement between two equal-length integer sequences."""
    yt, yp = _validate_sequences(
        y_true_sequence, y_pred_sequence,
        name="y_true_sequence/y_pred_sequence",
    )
    return float((yt == yp).mean())


def routine_similarity_score(
    pred_sequence,
    template_sequence,
    mode: str = "positional",
    n_classes: Optional[int] = None,
) -> float:
    """
    Similarity between a predicted routine and a cluster template.

    Parameters
    ----------
    mode : {'positional', 'frequency'}
        'positional' — fraction of time slots that match exactly.
        'frequency'  — cosine similarity over per-activity count histograms;
                       order-insensitive, captures "same mix of activities".
    n_classes : int, optional
        Required only in 'frequency' mode if it cannot be inferred from data.

    Returns 0.0 when either histogram is the zero vector (degenerate cosine).
    """
    pred, tmpl = _validate_sequences(
        pred_sequence, template_sequence,
        name="pred_sequence/template_sequence",
    )
    if mode == "positional":
        return float((pred == tmpl).mean())
    if mode == "frequency":
        K = n_classes if n_classes is not None else int(max(pred.max(), tmpl.max()) + 1)
        if K <= 0:
            raise ValueError("Could not infer a valid n_classes for 'frequency' mode.")
        p_hist = np.bincount(pred, minlength=K).astype(np.float64)
        t_hist = np.bincount(tmpl, minlength=K).astype(np.float64)
        denom = np.linalg.norm(p_hist) * np.linalg.norm(t_hist)
        if denom == 0.0:
            return 0.0
        return float(np.dot(p_hist, t_hist) / denom)
    raise ValueError(f"Unknown mode '{mode}'. Use 'positional' or 'frequency'.")


def deviation_reduction(
    original_sequence,
    pred_sequence,
    template_sequence,
) -> float:
    """
    How much the prediction reduces deviation from a template, relative to the
    original sequence. Uses Hamming distance (number of differing positions).

        d_orig = hamming(original, template)
        d_pred = hamming(pred,     template)
        deviation_reduction = (d_orig - d_pred) / d_orig

    Interpretation
    --------------
    1.0  : prediction matches the template exactly
    0.0  : prediction is no closer to the template than the original was
    < 0  : prediction is *further* from the template than the original

    Returns 0.0 when the original already matched the template (nothing to reduce).
    """
    orig, pred, tmpl = _validate_sequences(
        original_sequence, pred_sequence, template_sequence,
        name="original/pred/template_sequence",
    )
    d_orig = int((orig != tmpl).sum())
    d_pred = int((pred != tmpl).sum())
    if d_orig == 0:
        return 0.0
    return float((d_orig - d_pred) / d_orig)


# ---------------------------------------------------------------------------
# Aggregate entry point
# ---------------------------------------------------------------------------

def evaluate_all(
    y_true=None,
    y_scores=None,
    ks: Sequence[int] = (1, 3, 5, 10),
    pred_sequence=None,
    true_sequence=None,
    original_sequence=None,
    template_sequence=None,
    n_classes: Optional[int] = None,
) -> dict:
    """
    Run every metric whose inputs are provided. Missing inputs are skipped, so
    this is safe to call as your teammates' outputs evolve. Useful as a single
    call from notebooks once a model produces all of (y_scores, sequences).
    """
    results: dict = {}

    if y_true is not None and y_scores is not None:
        results.update(evaluate_ranking(y_true, y_scores, ks=ks))

    if true_sequence is not None and pred_sequence is not None:
        results["sequence_match"] = sequence_match_score(true_sequence, pred_sequence)

    if pred_sequence is not None and template_sequence is not None:
        results["routine_similarity_positional"] = routine_similarity_score(
            pred_sequence, template_sequence, mode="positional"
        )
        results["routine_similarity_frequency"] = routine_similarity_score(
            pred_sequence, template_sequence, mode="frequency", n_classes=n_classes
        )

    if (original_sequence is not None
            and pred_sequence is not None
            and template_sequence is not None):
        results["deviation_reduction"] = deviation_reduction(
            original_sequence, pred_sequence, template_sequence
        )

    return results