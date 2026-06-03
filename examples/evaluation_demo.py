"""
Smoke test / demo for Phase 5 metrics and Phase 6 plotting helpers.

Uses synthetic data only — does NOT depend on the real preprocessing, training,
or scoring pipeline. Safe to run before teammates' modules are populated.

Run from the repo root:
    python examples/evaluation_demo.py

It writes a handful of PNGs to examples/demo_outputs/.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless: don't try to open windows during the demo
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.eval.evaluation import (  # noqa: E402
    deviation_reduction,
    evaluate_all,
    evaluate_ranking,
    hit_rate_at_k,
    ndcg_at_k,
    next_activity_accuracy,
    routine_similarity_score,
    sequence_match_score,
)
from src.analysis.visualization import (  # noqa: E402
    plot_ablation_comparison,
    plot_error_by_activity,
    plot_error_by_time_slot,
    plot_model_comparison,
    plot_score_distribution,
    plot_template_heatmap,
)
from src.utils.activity_map import CATEGORIES as ACTIVITY_LABELS  # noqa: E402


N_SAMPLES = 200
N_CLASSES = len(ACTIVITY_LABELS)  # 11 for this project
N_SLOTS = 48
N_CLUSTERS = 5


def main() -> None:
    rng = np.random.default_rng(42)

    # ---- Ranking-style inputs ------------------------------------------------
    y_true = rng.integers(0, N_CLASSES, size=N_SAMPLES)
    y_scores = rng.normal(size=(N_SAMPLES, N_CLASSES))
    # Inject signal so the "model" beats random.
    y_scores[np.arange(N_SAMPLES), y_true] += 1.5

    print("=== Ranking metrics ===")
    print(f"accuracy:     {next_activity_accuracy(y_true, y_scores):.3f}")
    print(f"hit_rate@5:   {hit_rate_at_k(y_true, y_scores, k=5):.3f}")
    print(f"ndcg@5:       {ndcg_at_k(y_true, y_scores, k=5):.3f}")
    print("evaluate_ranking:")
    for k, v in evaluate_ranking(y_true, y_scores).items():
        print(f"  {k}: {v:.3f}")

    # ---- Sequence-style inputs ----------------------------------------------
    template = rng.integers(0, N_CLASSES, size=N_SLOTS)
    original = template.copy()
    flip_idx = rng.choice(N_SLOTS, size=20, replace=False)
    original[flip_idx] = rng.integers(0, N_CLASSES, size=20)
    pred = original.copy()
    fix_idx = rng.choice(flip_idx, size=12, replace=False)
    pred[fix_idx] = template[fix_idx]  # the "model" fixes 12 of 20 deviations

    print("\n=== Sequence metrics ===")
    print(f"sequence_match:                "
          f"{sequence_match_score(template, pred):.3f}")
    print(f"routine_similarity positional: "
          f"{routine_similarity_score(pred, template, mode='positional'):.3f}")
    print(f"routine_similarity frequency:  "
          f"{routine_similarity_score(pred, template, mode='frequency', n_classes=N_CLASSES):.3f}")
    print(f"deviation_reduction:           "
          f"{deviation_reduction(original, pred, template):.3f}")

    print("\n=== evaluate_all (combined) ===")
    combined = evaluate_all(
        y_true=y_true, y_scores=y_scores,
        pred_sequence=pred, true_sequence=template,
        original_sequence=original, template_sequence=template,
        n_classes=N_CLASSES,
    )
    for k, v in combined.items():
        if isinstance(v, dict):
            inner = ", ".join(
                f"{kk}={'na' if vv is None else f'{vv:.2f}'}"
                for kk, vv in v.items()
            )
            print(f"  {k}: {{ {inner} }}")
        else:
            print(f"  {k}: {v:.3f}")

    # ---- Plots ---------------------------------------------------------------
    out_dir = os.path.join(HERE, "demo_outputs")
    os.makedirs(out_dir, exist_ok=True)

    templates = rng.integers(0, N_CLASSES, size=(N_CLUSTERS, N_SLOTS))
    plot_template_heatmap(
        templates, activity_labels=ACTIVITY_LABELS,
        title="Cluster templates (demo)",
        save_path=os.path.join(out_dir, "templates.png"),
    )

    composite = rng.normal(loc=0.6, scale=0.15, size=400)
    cluster_ids = rng.integers(0, N_CLUSTERS, size=400)
    plot_score_distribution(
        composite, cluster_ids=cluster_ids,
        title="Composite scores per cluster (demo)",
        save_path=os.path.join(out_dir, "scores.png"),
    )

    yt_seq = rng.integers(0, N_CLASSES, size=N_SAMPLES)
    yp_seq = yt_seq.copy()
    swap = rng.random(N_SAMPLES) < 0.35
    yp_seq[swap] = rng.integers(0, N_CLASSES, size=swap.sum())
    slots = rng.integers(0, N_SLOTS, size=N_SAMPLES)

    plot_error_by_activity(
        yt_seq, yp_seq, activity_labels=ACTIVITY_LABELS,
        save_path=os.path.join(out_dir, "error_by_activity.png"),
    )
    plot_error_by_time_slot(
        slots, yt_seq, yp_seq,
        save_path=os.path.join(out_dir, "error_by_time_slot.png"),
    )

    # Fake ablation results: same scores perturbed by progressively more noise.
    results = {
        "GRU4Rec_lambda=0":   evaluate_ranking(y_true, y_scores),
        "GRU4Rec_lambda=0.5": evaluate_ranking(
            y_true, y_scores + rng.normal(scale=0.05, size=y_scores.shape)),
        "GRU4Rec_lambda=1.0": evaluate_ranking(
            y_true, y_scores + rng.normal(scale=0.10, size=y_scores.shape)),
    }
    plot_model_comparison(
        results, metric_name="ndcg@5",
        save_path=os.path.join(out_dir, "comparison_ndcg5.png"),
    )

    # Ablation: lambda=0 (fidelity only) vs lambda>0 (CE + KL).
    ablation = {
        "lambda=0":   results["GRU4Rec_lambda=0"],
        "lambda=0.5": results["GRU4Rec_lambda=0.5"],
        "lambda=1.0": results["GRU4Rec_lambda=1.0"],
    }
    plot_ablation_comparison(
        ablation,
        metric_names=["accuracy", "hit_rate@5", "ndcg@5"],
        title="Ablation: fidelity-only vs CE+KL (synthetic)",
        save_path=os.path.join(out_dir, "ablation_lambda.png"),
    )

    print(f"\nFigures written to: {out_dir}")


if __name__ == "__main__":
    main()