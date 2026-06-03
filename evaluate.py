"""
Entry point for Phase 5 evaluation against real model outputs.

Looks for saved predictions and (optionally) cluster templates / original
sequences. Runs every metric whose inputs are present and prints a JSON
report. If a required upstream artifact is missing, prints a clear pointer
to what teammates still need to produce.

Expected on-disk layout (relative to repo root):

    data/processed/
        sequences.pkl              # output of preprocess.py (dict TUCASEID -> (48,) int)
        predictions.npz            # produced by training/eval script (see below)

`predictions.npz` is the canonical hand-off from the training side. It must
contain at minimum:

    y_true       : int array, shape (n_samples,)              labels in [0, 11)
    y_scores     : float array, shape (n_samples, 11)         logits or probabilities

It may also include any of:

    time_slots         : int array (n_samples,)         slot index per sample (0..47)
    user_ids           : int array (n_samples,)         user index per sample
    routine_targets    : int array (n_samples,)         optimal-template activity per sample
    pred_sequence      : int array (48,)                model's predicted day for a user
    true_sequence      : int array (48,)                ground-truth day for that user
    original_sequence  : int array (48,)                original (un-fixed) day for deviation_reduction
    template_sequence  : int array (48,)                cluster template for that user

Run:

    PYTHONPATH=. python3 evaluate.py
    PYTHONPATH=. python3 evaluate.py --predictions path/to/file.npz --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from src.eval.evaluation import (
    evaluate_alignment,
    evaluate_all,
    evaluate_ranking,
    per_class_accuracy,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PRED_PATH = REPO_ROOT / "data" / "processed" / "predictions.npz"


def _load_npz(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _missing_files_message(pred_path: Path) -> str:
    return (
        "Cannot run real evaluation yet — required artifact(s) missing:\n"
        f"  - predictions file: {pred_path}\n"
        "\n"
        "Expected contents (numpy .npz):\n"
        "  y_true   : shape (n_samples,)        int labels in [0, 11)\n"
        "  y_scores : shape (n_samples, 11)     float logits/probabilities\n"
        "\n"
        "This file is produced by the training/evaluation step. Teammates working\n"
        "on src/training/train.py need to save the test-set predictions in that\n"
        "format. Example skeleton:\n"
        "\n"
        "    np.savez(\n"
        "        'data/processed/predictions.npz',\n"
        "        y_true=y_true,        # (N,)\n"
        "        y_scores=y_scores,    # (N, 11)\n"
        "        time_slots=time_slots # optional, for per-slot analysis\n"
        "    )\n"
        "\n"
        "Until that file exists, run the synthetic demo instead:\n"
        "    PYTHONPATH=. python3 examples/evaluation_demo.py\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 5 evaluation.")
    parser.add_argument(
        "--predictions", type=Path, default=DEFAULT_PRED_PATH,
        help=f"Path to .npz with y_true and y_scores (default: {DEFAULT_PRED_PATH})",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Optional path to write the metrics report as JSON.",
    )
    parser.add_argument(
        "--ks", type=int, nargs="+", default=[1, 3, 5, 10],
        help="k values for hit_rate@k and ndcg@k (default: 1 3 5 10).",
    )
    args = parser.parse_args()

    preds = _load_npz(args.predictions)
    if preds is None:
        print(_missing_files_message(args.predictions))
        return 1

    if "y_true" not in preds or "y_scores" not in preds:
        print(
            f"Predictions file {args.predictions} is missing required keys "
            "'y_true' and/or 'y_scores'."
        )
        return 1

    y_true = np.asarray(preds["y_true"])
    y_scores = np.asarray(preds["y_scores"])

    if y_true.ndim != 1 or y_scores.ndim != 2 or y_scores.shape[0] != y_true.shape[0]:
        print(
            f"Predictions file {args.predictions} has malformed shapes: "
            f"y_true{y_true.shape} (expected 1-D) and "
            f"y_scores{y_scores.shape} (expected 2-D with matching first dim)."
        )
        return 1

    results: dict = {
        "predictions_path": str(args.predictions),
        "n_samples": int(y_true.shape[0]),
        "n_classes": int(y_scores.shape[1]),
    }
    results.update(evaluate_ranking(y_true, y_scores, ks=tuple(args.ks)))
    results["per_class_accuracy"] = per_class_accuracy(y_true, y_scores)

    if "routine_targets" in preds:
        routine_targets = np.asarray(preds["routine_targets"])
        if routine_targets.shape == y_true.shape:
            alignment = evaluate_alignment(
                routine_targets, y_scores, ks=tuple(args.ks)
            )
            results.update(alignment)
            results["realism_minus_alignment_accuracy"] = (
                results["accuracy"] - alignment["alignment_accuracy"]
            )
        else:
            results["alignment_skipped"] = (
                f"routine_targets shape {routine_targets.shape} does not match "
                f"y_true shape {y_true.shape}."
            )
    else:
        results["alignment_skipped"] = (
            "No routine_targets in predictions file. Re-run prediction with a "
            "checkpoint whose training set was large enough to build routines "
            "so the runner emits them."
        )

    seq_kwargs = {}
    for key in ("pred_sequence", "true_sequence", "original_sequence", "template_sequence"):
        if key in preds:
            seq_kwargs[key] = preds[key]
    if seq_kwargs:
        seq_results = evaluate_all(
            y_true=None, y_scores=None,
            n_classes=int(y_scores.shape[1]),
            **seq_kwargs,
        )
        results.update(seq_results)
    else:
        results["sequence_metrics_skipped"] = (
            "No pred_sequence/true_sequence/template_sequence found in predictions file. "
            "These are optional; include them to compute sequence_match, "
            "routine_similarity, and deviation_reduction."
        )

    text = json.dumps(results, indent=2, default=float)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"\nWrote metrics report to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
