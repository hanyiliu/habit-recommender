# src/analysis/lambda_selection.py
"""Choose λ on the fidelity-vs-alignment tradeoff curve.

Selection rule (see design doc): λ* is the highest-alignment λ whose fidelity
stays within ``floor_frac`` (relative) of the λ=0 fidelity ceiling. The knee
(max-curvature point) is reported as a sanity check only — never used to pick λ*.
"""
from __future__ import annotations

import numpy as np


def _knee_lambda(results, primary_metric, alignment_metric):
    """λ of the point furthest from the chord joining the extreme-λ endpoints.

    Both axes are min-max normalized to [0, 1] first so the distance is not
    dominated by whichever metric has the larger numeric range.
    """
    pts = sorted(results, key=lambda r: r["lambda"])
    x = np.array([r[alignment_metric] for r in pts], dtype=float)
    y = np.array([r[primary_metric] for r in pts], dtype=float)

    def _norm(a):
        rng = a.max() - a.min()
        return (a - a.min()) / rng if rng > 0 else np.zeros_like(a)

    xn, yn = _norm(x), _norm(y)
    p1 = np.array([xn[0], yn[0]])
    p2 = np.array([xn[-1], yn[-1]])
    chord = p2 - p1
    length = float(np.hypot(chord[0], chord[1]))
    if length == 0.0:
        return pts[0]["lambda"]
    dists = []
    for i in range(len(pts)):
        v = np.array([xn[i], yn[i]]) - p1
        cross = chord[0] * v[1] - chord[1] * v[0]
        dists.append(abs(float(cross)) / length)
    return pts[int(np.argmax(dists))]["lambda"]


def select_lambda(
    results,
    primary_metric: str = "ndcg@5",
    alignment_metric: str = "alignment_ndcg@5",
    floor_frac: float = 0.05,
) -> dict:
    """Select λ* from per-λ results.

    Args:
        results: list of dicts, each with "lambda", primary_metric,
            alignment_metric. Must include a λ=0 row (the fidelity ceiling).
        primary_metric: fidelity key (vs the user's actual next activity).
        alignment_metric: alignment key (vs the routine template).
        floor_frac: max relative fidelity sacrifice from the λ=0 ceiling.

    Returns:
        dict with lambda_star, ceiling, floor, knee_lambda, candidates, rationale.
    """
    if not results:
        raise ValueError("results is empty.")
    if floor_frac < 0:
        raise ValueError(
            "floor_frac must be non-negative (a negative floor would exclude "
            "even the lambda=0 ceiling)."
        )
    base = [r for r in results if r["lambda"] == 0]
    if not base:
        raise ValueError(
            "results must include a lambda=0 row as the fidelity ceiling."
        )

    ceiling = float(base[0][primary_metric])
    floor = (1.0 - floor_frac) * ceiling
    candidates = [r for r in results if float(r[primary_metric]) >= floor]
    # Max alignment; tie-break toward the smaller λ (less nudging for the same
    # alignment is the more conservative, more credible choice).
    best = max(
        candidates,
        key=lambda r: (float(r[alignment_metric]), -float(r["lambda"])),
    )
    lambda_star = best["lambda"]
    knee = _knee_lambda(results, primary_metric, alignment_metric)

    rationale = (
        f"λ*={lambda_star}: highest {alignment_metric} "
        f"({best[alignment_metric]:.4f}) among λ whose {primary_metric} "
        f"({best[primary_metric]:.4f}) stays >= floor {floor:.4f} "
        f"(= {1 - floor_frac:.0%} of the λ=0 ceiling {ceiling:.4f}). "
        f"Knee (sanity check) at λ={knee}."
    )
    return {
        "lambda_star": lambda_star,
        "ceiling": ceiling,
        "floor": floor,
        "knee_lambda": knee,
        "candidates": [r["lambda"] for r in candidates],
        "rationale": rationale,
    }
