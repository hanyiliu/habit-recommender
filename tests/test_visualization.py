# tests/test_visualization.py
import matplotlib
matplotlib.use("Agg")  # headless; must precede any pyplot import

import pytest

from src.analysis.visualization import plot_alignment_tradeoff


def _results():
    return [
        {"lambda": 0.0, "ndcg@5": 0.85, "alignment_ndcg@5": 0.60},
        {"lambda": 0.5, "ndcg@5": 0.83, "alignment_ndcg@5": 0.75},
        {"lambda": 1.0, "ndcg@5": 0.81, "alignment_ndcg@5": 0.86},
    ]


def test_returns_fig_ax_and_labels():
    fig, ax = plot_alignment_tradeoff(
        _results(), selected_lambda=1.0, floor=0.8075,
    )
    assert fig is not None and ax is not None
    assert "alignment" in ax.get_xlabel().lower()
    # one annotation per λ point
    assert len(ax.texts) >= 3


def test_empty_results_raises():
    with pytest.raises(ValueError, match="empty"):
        plot_alignment_tradeoff([])
