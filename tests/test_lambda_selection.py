# tests/test_lambda_selection.py
import pytest

from src.analysis.lambda_selection import select_lambda


def _results():
    # fidelity (ndcg@5) decays with λ; alignment rises with λ.
    return [
        {"lambda": 0.0, "ndcg@5": 0.85, "alignment_ndcg@5": 0.60},
        {"lambda": 0.5, "ndcg@5": 0.83, "alignment_ndcg@5": 0.75},
        {"lambda": 1.0, "ndcg@5": 0.81, "alignment_ndcg@5": 0.86},
        {"lambda": 2.0, "ndcg@5": 0.74, "alignment_ndcg@5": 0.91},
    ]


def test_picks_highest_alignment_within_floor():
    # ceiling=0.85, floor=0.95*0.85=0.8075 → candidates λ∈{0,0.5,1.0}
    # (λ=2.0's 0.74 is below floor). Max alignment among them is λ=1.0.
    sel = select_lambda(_results(), floor_frac=0.05)
    assert sel["lambda_star"] == 1.0
    assert sel["ceiling"] == 0.85
    assert sel["floor"] == pytest.approx(0.8075)
    assert set(sel["candidates"]) == {0.0, 0.5, 1.0}
    assert isinstance(sel["rationale"], str) and sel["rationale"]


def test_tie_on_alignment_prefers_smaller_lambda():
    results = [
        {"lambda": 0.0, "ndcg@5": 0.85, "alignment_ndcg@5": 0.60},
        {"lambda": 0.5, "ndcg@5": 0.84, "alignment_ndcg@5": 0.80},
        {"lambda": 1.0, "ndcg@5": 0.82, "alignment_ndcg@5": 0.80},
    ]
    sel = select_lambda(results, floor_frac=0.05)
    assert sel["lambda_star"] == 0.5


def test_knee_is_a_grid_lambda():
    sel = select_lambda(_results(), floor_frac=0.05)
    assert sel["knee_lambda"] in {0.0, 0.5, 1.0, 2.0}


def test_requires_lambda_zero_baseline():
    results = [{"lambda": 0.5, "ndcg@5": 0.83, "alignment_ndcg@5": 0.75}]
    with pytest.raises(ValueError, match="lambda=0"):
        select_lambda(results)


def test_empty_results_raises():
    with pytest.raises(ValueError, match="empty"):
        select_lambda([])


def test_negative_floor_frac_raises():
    # A negative floor would push the floor above the ceiling, leaving no
    # candidates (even λ=0 fails) and an opaque max() error. Reject it up front.
    with pytest.raises(ValueError, match="floor_frac"):
        select_lambda(_results(), floor_frac=-0.1)


def test_duplicate_lambda_rows_pick_higher_alignment():
    results = [
        {"lambda": 0.0, "ndcg@5": 0.85, "alignment_ndcg@5": 0.60},
        {"lambda": 0.5, "ndcg@5": 0.83, "alignment_ndcg@5": 0.70},
        {"lambda": 0.5, "ndcg@5": 0.83, "alignment_ndcg@5": 0.78},
    ]
    sel = select_lambda(results, floor_frac=0.05)
    assert sel["lambda_star"] == 0.5
