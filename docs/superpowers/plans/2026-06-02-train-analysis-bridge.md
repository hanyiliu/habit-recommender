# Train → Analysis Bridge (Phase A: Ranking Handoff) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a trained checkpoint produce a `data/processed/predictions_<model>.npz` that `evaluate.py` already knows how to read, so the `preprocess → train → predict → evaluate` pipeline runs end-to-end for next-activity (ranking) metrics.

**Architecture:** A dedicated, model-agnostic prediction stage under `src/eval/predict/` (a shared `runner.py` core + a thin per-model CLI entry) loads a self-describing checkpoint, rebuilds the model, reconstructs the exact deterministic test split, runs forward inference, and serializes the flat `y_true / y_scores / time_slots / user_ids` arrays. Checkpoints become self-describing by having `Trainer.fit` store a `config` dict. Model-class lookup moves into a shared `src/models/registry.py` used by both training and prediction.

**Tech Stack:** Python 3.10+, PyTorch 2.x, NumPy. (Sequence-level "Regime B" metrics — autoregressive rollout, per-user routine templates, population aggregation — are intentionally **out of scope** here and tracked as a separate future plan; see the final section.)

> **Environment note for the executor:** All tests require PyTorch. Run `pip install -r requirements.txt` first. The doc-authoring sandbox could not install torch, so these tests have not been executed here — run them in a torch-capable environment.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/models/registry.py` (create) | `get_model_class(name)` — the single name→class lookup, shared by train + predict |
| `train_main.py` (modify) | use the shared registry; build and pass a `config` dict to `Trainer` |
| `src/training/train.py` (modify) | `Trainer` accepts `config` and stores it in the checkpoint |
| `src/eval/predict/__init__.py` (create) | package marker (empty) |
| `src/eval/predict/runner.py` (create) | model-agnostic core: load checkpoint, run inference, save npz, orchestrate |
| `src/eval/predict/gru4rec.py` (create) | thin CLI entry: `python -m src.eval.predict.gru4rec` |
| `tests/test_registry.py` (create) | registry lookup behavior |
| `tests/test_training.py` (modify) | checkpoint now carries `config` |
| `tests/test_predict.py` (create) | runner shapes/determinism + checkpoint round-trip |
| `README.md` (modify) | document the `predict` step in the E2E flow |

Decomposition rationale: all rec models share `forward(sequences, user_ids) -> (B, n_activities)`, so the inference loop belongs in one place (`runner.py`); per-model files exist only as discoverable entry points and stay ~15 lines. The registry is extracted so training and prediction never drift on how a model name resolves.

---

## Task 1: Shared model registry

**Files:**
- Create: `src/models/registry.py`
- Create: `tests/test_registry.py`
- Modify: `train_main.py` (remove local `_load_model_class`, import the shared one)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry.py
import pytest

from src.models.registry import get_model_class
from src.models.gru4rec import GRU4Rec


def test_get_model_class_gru4rec():
    assert get_model_class("gru4rec") is GRU4Rec


def test_unknown_model_raises_value_error():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_class("nope")


def test_unimplemented_ablation_raises_module_not_found():
    # LSTMRec / TransformerRec are planned but not yet implemented.
    with pytest.raises(ModuleNotFoundError, match="not yet implemented"):
        get_model_class("lstm")
    with pytest.raises(ModuleNotFoundError, match="not yet implemented"):
        get_model_class("transformer")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.models.registry'`

- [ ] **Step 3: Write minimal implementation**

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

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_registry.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Refactor train_main.py to use the registry**

In `train_main.py`, delete the entire `_load_model_class` function. Replace its import block and call site:

Remove:
```python
from src.models.gru4rec import GRU4Rec
```
Add (with the other `src...` imports):
```python
from src.models.registry import get_model_class
```
Change the call site in `main()` from:
```python
    ModelClass = _load_model_class(args.model)
```
to:
```python
    ModelClass = get_model_class(args.model)
```

- [ ] **Step 6: Verify nothing else references the old function**

Run: `grep -rn "_load_model_class" .`
Expected: no matches.
Run: `python -c "import train_main"`
Expected: no error.

- [ ] **Step 7: Commit**

```bash
git add src/models/registry.py tests/test_registry.py train_main.py
git commit -m "refactor: extract shared model registry"
```

---

## Task 2: Self-describing checkpoints

`Trainer.fit` currently saves `model_state / optimizer_state / scheduler_state / epoch / val_loss`. Add an optional `config` dict so the checkpoint records how to rebuild the model and reproduce the test split.

**Files:**
- Modify: `src/training/train.py` (`Trainer.__init__`, `Trainer.fit`)
- Modify: `train_main.py` (`main` builds and passes `config`)
- Modify: `tests/test_training.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_training.py`:

```python
def test_fit_saves_config(tmp_path):
    model  = GRU4Rec(n_users=5)
    loader = _make_toy_loader()
    ckpt   = str(tmp_path / "best.pt")
    cfg    = {"model": "gru4rec", "model_kwargs": {"n_users": 5}, "window": 8,
              "val_frac": 0.15, "test_frac": 0.15, "seed": 42,
              "k_routines": 10, "n_classes": 11}
    trainer = Trainer(model, loader, loader, config=cfg)
    trainer.fit(n_epochs=1, checkpoint_path=ckpt)
    saved = torch.load(ckpt, weights_only=False)
    assert saved["config"] == cfg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_training.py::test_fit_saves_config -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'config'`

- [ ] **Step 3: Write minimal implementation**

In `src/training/train.py`, add `config` to `__init__` (place the new parameter last, after `device`):
```python
        device: str = "cpu",
        config: dict | None = None,
    ):
```
and store it (add alongside the other assignments in `__init__`):
```python
        self.config       = config
```
In `fit`, add `"config": self.config` to the saved checkpoint dict (alongside the existing keys):
```python
                torch.save(
                    {
                        "epoch":            epoch,
                        "model_state":      self.model.state_dict(),
                        "optimizer_state":  self.optimizer.state_dict(),
                        "scheduler_state":  self.scheduler.state_dict(),
                        "val_loss":         val_loss,
                        "config":           self.config,
                    },
                    checkpoint_path,
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_training.py -v`
Expected: PASS (all existing training tests + `test_fit_saves_config`). Existing tests pass `config` implicitly as `None`, which is fine.

- [ ] **Step 5: Build and pass config in train_main.py**

In `train_main.py` `main()`, after the model is constructed (`model = ModelClass(...)`) and before `trainer = Trainer(...)`, insert:
```python
    config = {
        "model":        args.model,
        "model_kwargs": {"n_users": len(sequences)},  # only non-default ctor arg today
        "window":       args.window,
        "val_frac":     args.val_frac,
        "test_frac":    args.test_frac,
        "seed":         args.seed,
        "k_routines":   args.k_routines,
        "n_classes":    11,
    }
```
Then add `config=config` to the `Trainer(...)` call (after `device=args.device`).

- [ ] **Step 6: Verify train_main still imports**

Run: `python -c "import train_main"`
Expected: no error.

- [ ] **Step 7: Commit**

```bash
git add src/training/train.py train_main.py tests/test_training.py
git commit -m "feat: store model+split config in training checkpoint"
```

---

## Task 3: Prediction runner (model-agnostic core)

**Files:**
- Create: `src/eval/predict/__init__.py` (empty file)
- Create: `src/eval/predict/runner.py`
- Create: `tests/test_predict.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_predict.py
import numpy as np
import pytest
import torch

from src.models.gru4rec import GRU4Rec
from src.eval.predict.runner import (
    run_ranking_predictions, save_predictions, load_checkpoint,
    predict_from_checkpoint,
)


def _fake_sequences(n: int = 20, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {i: rng.integers(0, 11, size=48, dtype=np.int8) for i in range(n)}


def _config(n_users: int, window: int = 24) -> dict:
    return {"model": "gru4rec", "model_kwargs": {"n_users": n_users},
            "window": window, "val_frac": 0.15, "test_frac": 0.15,
            "seed": 42, "k_routines": 10, "n_classes": 11}


def test_run_ranking_predictions_shapes_and_ranges():
    seqs = _fake_sequences(20)
    cfg = _config(len(seqs), window=24)
    model = GRU4Rec(n_users=len(seqs))
    out = run_ranking_predictions(model, seqs, cfg, batch_size=16)

    assert set(out) == {"y_true", "y_scores", "time_slots", "user_ids"}
    n = out["y_true"].shape[0]
    # 3 test users (20 * 0.15) x (48 - 24) windows = 72
    assert n == 3 * (48 - 24)
    assert out["y_scores"].shape == (n, 11)
    assert out["time_slots"].min() >= 24 and out["time_slots"].max() <= 47
    assert out["user_ids"].min() >= 0 and out["user_ids"].max() < len(seqs)


def test_run_ranking_predictions_deterministic():
    seqs = _fake_sequences(20)
    cfg = _config(len(seqs))
    model = GRU4Rec(n_users=len(seqs))
    a = run_ranking_predictions(model, seqs, cfg, batch_size=16)
    b = run_ranking_predictions(model, seqs, cfg, batch_size=16)
    assert np.array_equal(a["y_true"], b["y_true"])
    assert np.array_equal(a["time_slots"], b["time_slots"])
    assert np.array_equal(a["user_ids"], b["user_ids"])


def test_save_predictions_roundtrip(tmp_path):
    arrays = {"y_true": np.zeros(4, np.int64),
              "y_scores": np.zeros((4, 11), np.float32),
              "time_slots": np.arange(4, dtype=np.int64),
              "user_ids": np.arange(4, dtype=np.int64)}
    out = tmp_path / "p.npz"
    save_predictions(arrays, str(out), "gru4rec")
    with np.load(out) as d:
        assert str(d["model"]) == "gru4rec"
        assert d["y_scores"].shape == (4, 11)


def test_load_checkpoint_requires_config(tmp_path):
    ckpt = tmp_path / "bad.pt"
    torch.save({"model_state": {}}, ckpt)  # no 'config'
    with pytest.raises(ValueError, match="no 'config'"):
        load_checkpoint(str(ckpt))


def test_predict_from_checkpoint_end_to_end(tmp_path):
    seqs = _fake_sequences(20)
    import pickle
    seq_path = tmp_path / "sequences.pkl"
    seq_path.write_bytes(pickle.dumps(seqs))

    model = GRU4Rec(n_users=len(seqs))
    ckpt = tmp_path / "best.pt"
    torch.save({"model_state": model.state_dict(),
                "config": _config(len(seqs))}, ckpt)

    out = tmp_path / "predictions_gru4rec.npz"
    predict_from_checkpoint(str(ckpt), str(seq_path), str(out))
    with np.load(out) as d:
        assert {"y_true", "y_scores", "time_slots", "user_ids"} <= set(d.files)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_predict.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.eval.predict'`

- [ ] **Step 3: Write minimal implementation**

Create the empty package marker:
```bash
: > src/eval/predict/__init__.py
```

Create `src/eval/predict/runner.py`:
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
from src.models.registry import get_model_class


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load a checkpoint and require it to be self-describing (has 'config')."""
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
    user_ids (N,). The test split and example order are fully determined by
    config['seed'/'val_frac'/'test_frac'/'window'], so output is reproducible.
    """
    model = model.to(device).eval()
    window = config["window"]

    _, _, test_seqs = train_val_test_split(
        sequences, val_frac=config["val_frac"],
        test_frac=config["test_frac"], seed=config["seed"],
    )
    user_to_idx = build_user_mapping(sequences)

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

    # With shuffle=False, example i is row=i//wps, t=i%wps, predicted absolute
    # slot = t + window. (wps = num_slots - window = ds.windows_per_seq)
    wps = ds.windows_per_seq
    time_slots = (np.arange(len(ds)) % wps + window).astype(np.int64)

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


def predict_from_checkpoint(checkpoint_path: str, sequences_path: str,
                            out_path: str | None = None,
                            batch_size: int = 256, device: str = "cpu") -> str:
    """Rebuild the model from a self-describing checkpoint, run inference, save npz."""
    ckpt = load_checkpoint(checkpoint_path)
    config = ckpt["config"]
    model_name = config["model"]

    model = get_model_class(model_name)(**config["model_kwargs"])
    model.load_state_dict(ckpt["model_state"])

    sequences = load_sequences(sequences_path)
    arrays = run_ranking_predictions(
        model, sequences, config, batch_size=batch_size, device=device,
    )
    out_path = out_path or f"data/processed/predictions_{model_name}.npz"
    save_predictions(arrays, out_path, model_name)
    return out_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_predict.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/eval/predict/__init__.py src/eval/predict/runner.py tests/test_predict.py
git commit -m "feat: add model-agnostic prediction runner"
```

---

## Task 4: GRU4Rec prediction entry point

**Files:**
- Create: `src/eval/predict/gru4rec.py`

The end-to-end behavior is already covered by `test_predict_from_checkpoint_end_to_end` (Task 3), which exercises the only logic this entry point contains. This task adds the discoverable CLI wrapper.

- [ ] **Step 1: Write the entry point**

```python
# src/eval/predict/gru4rec.py
"""Emit predictions.npz for a trained checkpoint (any model; name read from config).

Usage:
    python -m src.eval.predict.gru4rec \
        --checkpoint checkpoints/best.pt \
        --sequences data/processed/sequences.pkl \
        --out data/processed/predictions_gru4rec.npz
"""
import argparse

from src.eval.predict.runner import predict_from_checkpoint


def main():
    p = argparse.ArgumentParser(description="Predict with a trained checkpoint")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--sequences",  default="data/processed/sequences.pkl")
    p.add_argument("--out",        default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device",     default="cpu")
    args = p.parse_args()
    predict_from_checkpoint(
        args.checkpoint, args.sequences, args.out,
        batch_size=args.batch_size, device=args.device,
    )


if __name__ == "__main__":
    main()
```

> The body is model-agnostic (model name comes from the checkpoint `config`), so LSTM/Transformer entries added later are near-identical wrappers calling the same `predict_from_checkpoint`.

- [ ] **Step 2: Verify the CLI wiring imports and parses**

Run: `python -m src.eval.predict.gru4rec --help`
Expected: argparse usage text prints, exit 0.

- [ ] **Step 3: Commit**

```bash
git add src/eval/predict/gru4rec.py
git commit -m "feat: add GRU4Rec prediction entry point"
```

---

## Task 5: Document the predict step and verify E2E

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the README pipeline section**

In `README.md`, add the `predict` step between training and evaluation, and move the "predictions.npz handoff" item out of the known-gaps list. The runnable flow becomes:
```bash
python preprocess.py
python train_main.py --epochs 5
python -m src.eval.predict.gru4rec --checkpoint checkpoints/best.pt
python evaluate.py --predictions data/processed/predictions_gru4rec.npz
```
Update the known-gaps section to keep only: LSTM/Transformer ablations, full-day autoregressive rollout for sequence-level metrics (Regime B), and CI.

- [ ] **Step 2: Run the full test suite**

Run: `pytest tests/ -v`
Expected: PASS — all of `test_dataset.py`, `test_evaluation.py`, `test_training.py`, `test_registry.py`, `test_predict.py`.

- [ ] **Step 3: End-to-end smoke (requires the real data + torch)**

Run:
```bash
python preprocess.py
python train_main.py --epochs 2
python -m src.eval.predict.gru4rec --checkpoint checkpoints/best.pt
python evaluate.py --predictions data/processed/predictions_gru4rec.npz
```
Expected: `evaluate.py` prints a JSON report containing `accuracy`, `hit_rate@k`, `ndcg@k`, and `per_class_accuracy` — and does **not** print the "Cannot run real evaluation yet — required artifact(s) missing" message.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document predict step in E2E pipeline"
```

---

## Phase A Definition of Done

`evaluate.py` produces real ranking + per-class metrics from a trained checkpoint with zero manual flag-matching, and the model-comparison / per-slot / per-activity plots in `examples/` can be driven from a real `predictions_<model>.npz`. Full `pytest tests/` is green.

---

## Out of Scope — Future Plan: Regime B (sequence & routine metrics)

Not part of this plan. Track as a separate spec/plan once Phase A lands (and ideally after the LSTM/Transformer ablation models exist). It will add:

- **Autoregressive rollout** (`rollout(model, seed_context, window, n_slots=48)` in `runner.py`) to produce a full 48-slot `pred_sequence` per user — open decision: greedy vs sampled decoding.
- **Per-user routine templates** via the existing `RoutineMatcher` — open decision: serialize `routines` alongside the checkpoint vs. rebuild from the train split using stored `k_routines`/`seed`.
- **Population aggregation** — extend `src/eval/evaluation.py`'s sequence metrics (today single-day) to accept batched `(M, 48)` arrays; teach `evaluate.py` to read the batched keys.
- **Extended npz** — `pred_sequences`, `original_sequences`, `template_sequences`, matching `user_ids`.
- **Per-model entries** `src/eval/predict/{lstm,transformer}.py` (~15 lines each, reusing `predict_from_checkpoint`).

---

## Self-Review

- **Spec coverage:** Brainstorm decisions → tasks: phased/Regime-A-only → whole plan + explicit out-of-scope section ✓; dedicated `src/eval/predict/` → Tasks 3–4 ✓; shared core + thin entry → `runner.py` (Task 3) + `gru4rec.py` (Task 4) ✓; self-describing checkpoint config → Task 2 ✓; per-model `predictions_<model>.npz` → `save_predictions`/`predict_from_checkpoint` default ✓; registry for ablation-readiness → Task 1 ✓.
- **Type/name consistency:** `get_model_class` (Tasks 1, 3); `config` keys `model`/`model_kwargs`/`window`/`val_frac`/`test_frac`/`seed`/`k_routines`/`n_classes` defined in Task 2 and consumed identically in Task 3; `load_checkpoint` / `run_ranking_predictions` / `save_predictions` / `predict_from_checkpoint` signatures match between `runner.py` (Task 3), its tests (Task 3), and `gru4rec.py` (Task 4). `HabitDataset(arr, window_size=…, user_ids=…)` 3-tuple matches the merged `src/data/preprocessing/dataset.py`. `evaluate.py` already reads `y_true/y_scores/time_slots/user_ids` — no change needed.
- **Placeholder scan:** No TBD/TODO; every code step contains complete code; every run step states expected output.
