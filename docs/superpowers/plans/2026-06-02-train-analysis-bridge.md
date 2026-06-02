# Train → Analysis Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the gap between training and the analysis stage. Today `train_main.py` trains a model and prints test metrics but persists nothing for analysis, while `evaluate.py` expects an on-disk `predictions.npz` it never receives. This plan produces that artifact via a dedicated, ablation-aware prediction stage.

**Tech Stack:** Python 3.10+, PyTorch 2.x, NumPy.

---

## Background: why the bridge has two regimes

`evaluate.py` / `src/eval/evaluation.py` support **two distinct evaluation regimes** with very different costs:

- **Regime A — ranking / next-activity** (`evaluate_ranking`, `per_class_accuracy`): needs flat per-example arrays `y_true (N,)`, `y_scores (N, 11)`, optional `time_slots (N,)`, `user_ids (N,)`. These come directly from running the trained model over the test set — one prediction per sliding-window example. `train_main.evaluate_model` already computes the logits and discards them.
- **Regime B — sequence / routine** (`sequence_match_score`, `routine_similarity_score`, `deviation_reduction`): needs full 48-slot **day** sequences (`pred_sequence`, `original_sequence`, `template_sequence`). This requires autoregressive rollout (the model predicts one slot at a time) plus per-user routine templates, and `evaluate.py`'s sequence path currently only handles a *single* day, not a population.

**This plan implements Regime A now (Phase A) and specs Regime B as a documented follow-up (Phase B).**

## Design decisions (locked)

1. **Phased**: Phase A (ranking handoff) ships first and yields a working E2E `preprocess → train → predict → evaluate` run. Phase B (rollout + sequence metrics) is designed here but implemented later.
2. **Dedicated prediction stage** under `src/eval/predict/`, **not** inline in `train_main.py` — decouples inference from training so predictions can be regenerated for any checkpoint without retraining.
3. **Shared core + thin per-model entry points**: all rec models share the identical `forward(sequences, user_ids) -> (B, 11)` signature, so the inference loop (and later the rollout loop) lives once in `runner.py`; each model gets a ~10-line CLI entry. Ablation models (LSTMRec, TransformerRec) drop in cleanly.
4. **Self-describing checkpoints**: `Trainer.fit` stores a `config` dict (model name + constructor kwargs + data/split params + window) in the checkpoint so `predict` rebuilds the architecture and reproduces the exact test split with no guessing and no re-typed flags.
5. **Per-model output**: each entry writes `data/processed/predictions_<model>.npz` so ablation runs don't clobber each other; `evaluate.py --predictions …` points at the chosen file.

---

## File Map

| File | Responsibility | Phase |
|---|---|---|
| `src/models/registry.py` | `get_model_class(name)` — single model registry shared by train + predict | A |
| `src/training/train.py` | `Trainer.fit` saves a `config` dict into the checkpoint | A |
| `train_main.py` | builds + passes `config`; uses the shared registry | A |
| `src/eval/predict/__init__.py` | package marker | A |
| `src/eval/predict/runner.py` | `run_ranking_predictions(...)`, `save_predictions(...)`, `load_checkpoint_config(...)` (model-agnostic) | A |
| `src/eval/predict/gru4rec.py` | thin CLI entry: rebuild GRU4Rec from checkpoint, call runner, save npz | A |
| `tests/test_predict.py` | runner shape/determinism + checkpoint-config round-trip tests | A |
| `src/eval/predict/runner.py` (+`rollout`) | autoregressive full-day rollout; per-user templates | B |
| `src/eval/evaluation.py` | batched sequence metrics over a population | B |
| `src/eval/predict/{lstm,transformer}.py` | thin entries for ablation models | B (after models exist) |

---

## Phase A — Ranking handoff

### Task 1: Shared model registry

Extract model lookup out of `train_main.py` so both training and prediction resolve model classes the same way (and ablation models register in one place).

**Files:**
- Create: `src/models/registry.py`
- Edit: `train_main.py`

- [ ] **Step 1: Create the registry**

```python
# src/models/registry.py
"""Single source of truth for mapping a model name to its class."""
from src.models.gru4rec import GRU4Rec


def get_model_class(name: str):
    if name == "gru4rec":
        return GRU4Rec
    if name == "lstm":
        try:
            from src.models.lstm_rec import LSTMRec
            return LSTMRec
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "LSTMRec is not yet implemented. "
                "See docs/superpowers/plans/2026-05-26-ablation-models.md."
            ) from None
    if name == "transformer":
        try:
            from src.models.transformer_rec import TransformerRec
            return TransformerRec
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "TransformerRec is not yet implemented. "
                "See docs/superpowers/plans/2026-05-26-ablation-models.md."
            ) from None
    raise ValueError(f"Unknown model: {name}")
```

- [ ] **Step 2:** In `train_main.py`, delete the local `_load_model_class` and import `get_model_class` from `src.models.registry`; replace the call site.

- [ ] **Step 3: Commit** — `refactor: extract shared model registry`

### Task 2: Self-describing checkpoints

**Files:**
- Edit: `src/training/train.py`
- Edit: `train_main.py`

- [ ] **Step 1:** Add an optional `config: dict | None = None` parameter to `Trainer.__init__` (store as `self.config`). In `fit`, include it in the saved checkpoint dict as `"config": self.config`. Leave all existing keys intact (backward compatible — existing tests that don't pass a config still pass, the key is just `None`).

- [ ] **Step 2:** In `train_main.main`, build the config after parsing args and constructing the model, and pass it to `Trainer`:

```python
config = {
    "model": args.model,
    "model_kwargs": {"n_users": len(sequences)},  # only non-default ctor arg today
    "window": args.window,
    "val_frac": args.val_frac,
    "test_frac": args.test_frac,
    "seed": args.seed,
    "k_routines": args.k_routines,
    "n_classes": 11,
}
trainer = Trainer(model, train_loader, val_loader, lr=args.lr,
                  lambda_kl=args.lambda_kl, device=args.device, config=config)
```

> Rationale: `predict` needs `model` + `model_kwargs` to rebuild the architecture and load `state_dict`, and `window`/`val_frac`/`test_frac`/`seed` to reproduce the **exact** test split deterministically.

- [ ] **Step 3:** Update `tests/test_training.py::test_fit_saves_checkpoint` to assert `"config" in saved`. Add a case passing a config and asserting round-trip equality.

- [ ] **Step 4: Commit** — `feat: store model+split config in training checkpoint`

### Task 3: Prediction runner (shared core)

**Files:**
- Create: `src/eval/predict/__init__.py` (empty)
- Create: `src/eval/predict/runner.py`

- [ ] **Step 1: Implement the runner.** It is fully model-agnostic — it takes an already-constructed model and the config, rebuilds the deterministic test split, runs forward over it with `shuffle=False`, and returns the Regime-A arrays.

```python
# src/eval/predict/runner.py
"""Model-agnostic inference: turn a trained model + sequences into predictions.npz arrays."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.preprocessing.dataset import (
    HabitDataset, build_user_mapping, train_val_test_split,
)
from src.data.preprocessing.preprocessor import load_sequences


def load_checkpoint_config(checkpoint_path: str) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if ckpt.get("config") is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no 'config'. Re-train with the "
            "current train_main.py so the checkpoint is self-describing."
        )
    return ckpt


@torch.no_grad()
def run_ranking_predictions(model, sequences: dict, config: dict,
                            batch_size: int = 256, device: str = "cpu") -> dict:
    """Run the model over the deterministic test split; return Regime-A arrays.

    Returns dict with: y_true (N,), y_scores (N, n_classes), time_slots (N,),
    user_ids (N,).
    """
    model = model.to(device).eval()
    window = config["window"]

    _, _, test_seqs = train_val_test_split(
        sequences, val_frac=config["val_frac"],
        test_frac=config["test_frac"], seed=config["seed"],
    )
    user_to_idx = build_user_mapping(sequences)

    # Match train_main's per-split build, minus routines (not needed for ranking).
    arr = np.stack([test_seqs[k] for k in test_seqs])
    uids = np.array([user_to_idx[k] for k in test_seqs], dtype=np.int64)
    ds = HabitDataset(arr, window_size=window, user_ids=uids)  # 3-tuple (x, y, user_id)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    y_scores, y_true, user_ids = [], [], []
    for context, targets, batch_uids in loader:
        logits = model(context.to(device), batch_uids.to(device)).cpu()
        y_scores.append(logits)
        y_true.append(targets)
        user_ids.append(batch_uids)
    y_scores = torch.cat(y_scores).numpy()
    y_true = torch.cat(y_true).numpy()
    user_ids = torch.cat(user_ids).numpy()

    # time_slots: with shuffle=False the i-th example is row=i//wps, t=i%wps,
    # predicted absolute slot = t + window. (wps = num_slots - window)
    wps = ds.windows_per_seq
    n = len(ds)
    time_slots = (np.arange(n) % wps + window).astype(np.int64)

    return {
        "y_true": y_true.astype(np.int64),
        "y_scores": y_scores.astype(np.float32),
        "time_slots": time_slots,
        "user_ids": user_ids.astype(np.int64),
    }


def save_predictions(arrays: dict, out_path: str, model_name: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, model=np.array(model_name), **arrays)
    print(f"Saved {len(arrays['y_true'])} predictions to {out}")
```

- [ ] **Step 2: Commit** — `feat: add model-agnostic prediction runner`

### Task 4: GRU4Rec prediction entry

**Files:**
- Create: `src/eval/predict/gru4rec.py`

- [ ] **Step 1: Implement the thin entry.**

```python
# src/eval/predict/gru4rec.py
"""Emit predictions.npz for a trained GRU4Rec checkpoint.

Usage:
    python -m src.eval.predict.gru4rec \
        --checkpoint checkpoints/best.pt \
        --sequences data/processed/sequences.pkl \
        --out data/processed/predictions_gru4rec.npz
"""
import argparse

from src.data.preprocessing.preprocessor import load_sequences
from src.models.registry import get_model_class
from src.eval.predict.runner import (
    load_checkpoint_config, run_ranking_predictions, save_predictions,
)


def main():
    p = argparse.ArgumentParser(description="Predict with a trained checkpoint")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--sequences",  default="data/processed/sequences.pkl")
    p.add_argument("--out",        default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device",     default="cpu")
    args = p.parse_args()

    ckpt = load_checkpoint_config(args.checkpoint)
    config = ckpt["config"]
    model_name = config["model"]

    model = get_model_class(model_name)(**config["model_kwargs"])
    model.load_state_dict(ckpt["model_state"])

    sequences = load_sequences(args.sequences)
    arrays = run_ranking_predictions(
        model, sequences, config,
        batch_size=args.batch_size, device=args.device,
    )
    out = args.out or f"data/processed/predictions_{model_name}.npz"
    save_predictions(arrays, out, model_name)


if __name__ == "__main__":
    main()
```

> Note: the entry is intentionally generic — `model_name` comes from the checkpoint config, so the same body works for any model. The per-model file exists so each model has a discoverable entry point (`python -m src.eval.predict.<model>`); the LSTM/Transformer entries are near-identical and added in Phase B once those models exist.

- [ ] **Step 2: Commit** — `feat: add GRU4Rec prediction entry point`

### Task 5: Tests

**Files:**
- Create: `tests/test_predict.py`

- [ ] **Step 1:** Tests using a tiny synthetic sequences dict and a real `GRU4Rec` (small `n_users`):
  - `run_ranking_predictions` returns the four keys with consistent first dim `N`, `y_scores` shape `(N, 11)`, `time_slots` within `[window, 47]`, and `user_ids` drawn from the global mapping.
  - **Determinism**: two runs with the same config produce identical `y_true`/`time_slots`/`user_ids` (split is seeded).
  - `save_predictions` round-trips: `np.load` returns the same arrays plus `model`.
  - `load_checkpoint_config` raises a clear error when `config` is absent.
  - (torch is required; mirror the existing torch-dependent tests — they aren't run in the doc-authoring sandbox but run in a torch env.)

- [ ] **Step 2: Commit** — `test: cover prediction runner and checkpoint config`

### Task 6: Docs + end-to-end verification

- [ ] **Step 1:** Update `README.md`: add the `predict` step between train and evaluate, and move the predictions.npz item out of "known gaps":
  ```bash
  python preprocess.py
  python train_main.py --epochs 5
  python -m src.eval.predict.gru4rec --checkpoint checkpoints/best.pt
  python evaluate.py --predictions data/processed/predictions_gru4rec.npz
  ```
- [ ] **Step 2: Verify E2E** in a torch environment: the four commands above run clean and `evaluate.py` prints ranking + per-class metrics (no "missing artifact" message). `pytest tests/` green.
- [ ] **Step 3: Commit** — `docs: document predict step in E2E pipeline`

**Phase A definition of done:** `evaluate.py` produces real ranking/per-class metrics from a trained checkpoint with zero manual flag-matching, and the model-comparison/per-slot/per-activity plots in `examples/` can be driven from real `predictions_<model>.npz`.

---

## Phase B — Sequence & routine metrics (design only; implement later)

Implement after Phase A and (ideally) after the ablation models exist. Adds the full-day capability that unlocks `sequence_match_score`, `routine_similarity_score`, and `deviation_reduction`.

1. **Autoregressive rollout** — add `rollout(model, seed_context, window, n_slots=48, device)` to `runner.py`: seed with each test user's first `window` real slots, then predict slot-by-slot (greedy argmax — decision: greedy vs sampling), feeding each prediction back as context. Produces one `pred_sequence (48,)` per user. Shared so ablation models reuse it verbatim.
2. **Per-user templates** — the nearest routine template per user via the existing `RoutineMatcher`. Routines must be reachable at predict time: either (a) serialize `routines (R, 48)` alongside the checkpoint during training, or (b) rebuild them in predict from the train split using `build_routines` with the stored `k_routines`/`seed` (decision pending; (a) is more faithful, (b) avoids a new artifact). `template_sequence` = the full nearest-routine day; `original_sequence` = the user's real day; `true_sequence` = same real day for `sequence_match`.
3. **Population aggregation** — `evaluate.py`'s sequence path takes single sequences today. Extend `src/eval/evaluation.py` with batched variants (accept `(M, 48)` arrays and average), or add `evaluate_sequences(preds, originals, templates)` that loops and returns means. This is the one place Phase B touches existing eval code.
4. **Extend `predictions_<model>.npz`** with `pred_sequences (M, 48)`, `original_sequences (M, 48)`, `template_sequences (M, 48)`, and the corresponding `user_ids (M,)`; teach `evaluate.py` to consume the batched keys.
5. **Per-model entries** — add `src/eval/predict/lstm.py` and `transformer.py` (each ~10 lines, reusing the runner) once those models land.

**Open questions to resolve at Phase B kickoff:** greedy vs sampled decoding; serialize-vs-rebuild routines; whether `deviation_reduction`'s "original" should be the raw day or a partially-observed day. None block Phase A.
