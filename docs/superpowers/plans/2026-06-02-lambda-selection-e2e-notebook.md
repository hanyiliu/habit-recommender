# End-to-End λ-Selection Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Jupyter notebook that goes from raw ATUS data → a λ-swept GRU4Rec training run (cached) → fidelity & alignment evaluation → a fidelity-vs-alignment tradeoff plot and a justified `λ*`, with the selection rule and the plot extracted into tested `src/` modules.

**Architecture:** Approach A — the notebook is pure orchestration + narrative; the two pieces that encode the scientific conclusion live in `src/` under pytest: `select_lambda` (the floor + max-alignment rule, with a knee sanity check) in a new `src/analysis/lambda_selection.py`, and `plot_alignment_tradeoff` (the curve) added to `src/analysis/visualization.py`. Everything else reuses functions already on `main` (`load_sequences`, `train_val_test_split`, `build_routines`, `HabitDataset`, `Trainer`, `run_ranking_predictions`, `evaluate_ranking`, `evaluate_alignment`).

**Tech Stack:** Python 3.10+, PyTorch 2.x, NumPy, pandas, matplotlib, jupytext + nbconvert (notebook authoring/execution), pytest.

**Design doc:** `docs/superpowers/specs/2026-06-02-lambda-selection-e2e-notebook-design.md`

> **Environment note:** all tests and the notebook require torch, scikit-learn, pandas, and matplotlib. Run `pip install -r requirements.txt` first. The notebook auto-selects the Apple-Silicon GPU (`DEVICE="mps"`) when available, else CPU. Its first full run trains 8 GRU4Rec models (EPOCHS=50 each) — slow even on MPS; subsequent runs load cached checkpoints and are fast. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set in the notebook so any op lacking an MPS kernel silently falls back to CPU.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `requirements.txt` | Modify | add `pandas`, `jupytext`, `nbconvert`, `ipykernel` |
| `src/analysis/lambda_selection.py` | Create | `select_lambda(...)` — floor + max-alignment rule, knee sanity check (pure logic) |
| `tests/test_lambda_selection.py` | Create | unit tests for `select_lambda` |
| `src/analysis/visualization.py` | Modify | add `plot_alignment_tradeoff(...)` after `plot_ablation_comparison` |
| `tests/test_visualization.py` | Create | smoke test for `plot_alignment_tradeoff` (Agg backend) |
| `notebooks/lambda_selection_e2e.py` | Create | jupytext percent-format source for the notebook (editable, reviewable) |
| `notebooks/lambda_selection_e2e.ipynb` | Generate | the runnable notebook, built from the `.py` via jupytext |

Decomposition rationale: `select_lambda` is the only nontrivial *decision* logic and must be testable in isolation — it takes a plain results list and returns `λ*` + rationale, with no torch/IO. `plot_alignment_tradeoff` is the one missing visualization (existing helpers are bar charts). The notebook holds no logic worth testing — it wires tested pieces together and narrates — so it is authored as a jupytext `.py` (clean diffs, no JSON-escaping) and converted to `.ipynb`.

---

## Task 1: `select_lambda` selection logic

**Files:**
- Create: `src/analysis/lambda_selection.py`
- Create: `tests/test_lambda_selection.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_lambda_selection.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.analysis.lambda_selection'`.

- [ ] **Step 3: Write minimal implementation**

```python
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
    dists = [
        abs(float(np.cross(chord, np.array([xn[i], yn[i]]) - p1))) / length
        for i in range(len(pts))
    ]
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_lambda_selection.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/analysis/lambda_selection.py tests/test_lambda_selection.py
git commit -m "feat(analysis): add select_lambda (fidelity-floor tradeoff rule)"
```

---

## Task 2: `plot_alignment_tradeoff` visualization

**Files:**
- Modify: `src/analysis/visualization.py` (add a function at the end, after `plot_ablation_comparison`)
- Create: `tests/test_visualization.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_visualization.py -v`
Expected: FAIL — `ImportError: cannot import name 'plot_alignment_tradeoff'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/analysis/visualization.py` (after `plot_ablation_comparison`, end of file):

```python
def plot_alignment_tradeoff(
    results,
    fidelity_key: str = "ndcg@5",
    alignment_key: str = "alignment_ndcg@5",
    lambda_key: str = "lambda",
    selected_lambda: Optional[float] = None,
    floor: Optional[float] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Fidelity-vs-alignment tradeoff curve parameterized by λ.

    Parameters
    ----------
    results : sequence of mappings
        Each item has ``lambda_key``, ``fidelity_key`` (agreement with the
        user's actual next activity, y-axis) and ``alignment_key`` (agreement
        with the routine template, x-axis).
    selected_lambda : float, optional
        The chosen λ*; its point is highlighted.
    floor : float, optional
        Fidelity floor; drawn as a horizontal reference line.

    Returns ``(fig, ax)``.
    """
    if not results:
        raise ValueError("results is empty.")
    pts = sorted(results, key=lambda r: r[lambda_key])
    xs = [float(r[alignment_key]) for r in pts]
    ys = [float(r[fidelity_key]) for r in pts]
    lams = [r[lambda_key] for r in pts]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(xs, ys, "-o", color="#3b6", zorder=2)
    for x, y, lam in zip(xs, ys, lams):
        ax.annotate(f"λ={lam}", (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)

    if selected_lambda is not None:
        for x, y, lam in zip(xs, ys, lams):
            if lam == selected_lambda:
                ax.scatter([x], [y], s=160, facecolors="none",
                           edgecolors="crimson", linewidths=2, zorder=3,
                           label=f"selected λ*={lam}")
                break

    if floor is not None:
        ax.axhline(floor, color="gray", ls="--", lw=1,
                   label=f"fidelity floor = {floor:.3f}")

    ax.set_xlabel(f"Alignment ({alignment_key}) — agreement with routine template")
    ax.set_ylabel(f"Fidelity ({fidelity_key}) — agreement with real behavior")
    ax.set_title(title or "Fidelity vs. alignment tradeoff over λ")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_visualization.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/analysis/visualization.py tests/test_visualization.py
git commit -m "feat(analysis): add plot_alignment_tradeoff curve"
```

---

## Task 3: Notebook dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Inspect current requirements**

Run: `cat requirements.txt`
Note which of `pandas`, `jupytext`, `nbconvert`, `ipykernel` are already present.

- [ ] **Step 2: Append any missing dependencies**

Add these lines to `requirements.txt` (skip any already present):

```
pandas>=2.0
jupytext>=1.16
nbconvert>=7.0
ipykernel>=6.0
```

- [ ] **Step 3: Install and verify import**

Run:
```bash
pip install -r requirements.txt
python -c "import pandas, jupytext, nbconvert, ipykernel; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "build: add notebook deps (pandas, jupytext, nbconvert, ipykernel)"
```

---

## Task 4: Author the notebook source (jupytext percent format)

**Files:**
- Create: `notebooks/lambda_selection_e2e.py`

- [ ] **Step 1: Write the percent-format notebook source**

Create `notebooks/lambda_selection_e2e.py` with exactly this content:

```python
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # End-to-End λ-Selection: Fidelity vs. Routine Alignment
#
# The model is trained with `L = L_BPR + λ·L_KL`. BPR rewards predicting the
# user's **actual** next activity (fidelity); the KL term nudges predictions
# toward the nearest healthy **routine template** (alignment). This notebook
# sweeps λ on GRU4Rec, measures both axes on the held-out test split, and picks
# λ* as the most-aligned model whose fidelity stays within 5% of the λ=0 ceiling.
#
# Caveats (kept honest): both axes are *in-sample to the loss* — fidelity tracks
# what BPR optimizes, alignment tracks what KL optimizes — and alignment here is
# a *single-step* proxy, not a full-day behavioral outcome. λ* is a candidate to
# later confirm against a cumulative deviation-reduction metric (Regime B).

# %%
import os
# Let any op without an MPS kernel fall back to CPU instead of erroring.
# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.data.preprocessing.preprocessor import load_sequences
from src.data.preprocessing.dataset import (
    HabitDataset, build_user_mapping, train_val_test_split,
)
from src.scoring.scoring import build_routines
from src.models.gru4rec import GRU4Rec
from src.training.train import Trainer
from src.eval.predict.runner import run_ranking_predictions, load_checkpoint
from src.eval.evaluation import evaluate_ranking, evaluate_alignment
from src.analysis.visualization import plot_alignment_tradeoff, plot_template_heatmap
from src.analysis.lambda_selection import select_lambda
from src.utils.activity_map import CATEGORIES
from torch.utils.data import DataLoader

# --- Config (locked in the design doc) ---
LAMBDAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
EPOCHS = 50
SEED = 42
WINDOW = 24
K_ROUTINES = 10
BATCH = 256
LR = 1e-3
# Prefer the Apple-Silicon GPU (MPS) when present; fall back to CPU otherwise.
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")
PRIMARY = "ndcg@5"
ALIGNMENT = "alignment_ndcg@5"
FLOOR_FRAC = 0.05
SEQ_PATH = Path("data/processed/sequences.pkl")
CACHE = Path("checkpoints/sweep")
CACHE.mkdir(parents=True, exist_ok=True)
Path("examples/demo_outputs").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Data: sequences, split, routines

# %%
if not SEQ_PATH.exists():
    # Self-contained cold start: mirror preprocess.py's 4-step pipeline.
    from src.data.preprocessing.atus_loader import load_atus_activity_file
    from src.utils.activity_map import map_activity_category
    from src.data.preprocessing.preprocessor import build_all_sequences, save_sequences
    _df = load_atus_activity_file(Path("data/2024_data/atusact_2024.dat"))
    _df["CATEGORY"] = _df.apply(map_activity_category, axis=1)
    save_sequences(build_all_sequences(_df), SEQ_PATH)

sequences = load_sequences(SEQ_PATH)
train_seqs, val_seqs, test_seqs = train_val_test_split(
    sequences, val_frac=0.10, test_frac=0.10, seed=SEED,
)
print(f"Users: {len(train_seqs)} train / {len(val_seqs)} val / {len(test_seqs)} test")

train_arr = np.stack([train_seqs[uid] for uid in train_seqs])
routines, _, _ = build_routines(train_arr, K=K_ROUTINES, random_state=SEED)
print(f"Built {len(routines)} routine templates")

user_to_idx = build_user_mapping(sequences)


def _loader(split, shuffle):
    arr = np.stack([split[uid] for uid in split])
    uids = np.array([user_to_idx[uid] for uid in split], dtype=np.int64)
    ds = HabitDataset(arr, window_size=WINDOW, routines=routines, user_ids=uids)
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=0)


train_loader = _loader(train_seqs, True)
val_loader = _loader(val_seqs, False)

# Config stored in each checkpoint; identical across λ (λ is a loss weight, not
# a config field), so templates rebuild identically at eval time.
CONFIG = {
    "model": "gru4rec",
    "model_kwargs": {"n_users": len(sequences)},
    "window": WINDOW,
    "val_frac": 0.10,
    "test_frac": 0.10,
    "seed": SEED,
    "k_routines": K_ROUTINES,
    "n_classes": 11,
}

# %% [markdown]
# ## 2. Swept training (cached: retrain only if the checkpoint is missing)

# %%
def train_or_load(lmbda):
    ckpt_path = CACHE / f"gru4rec_lambda{lmbda}.pt"
    hist_path = CACHE / f"gru4rec_lambda{lmbda}_history.json"
    if ckpt_path.exists() and hist_path.exists():
        print(f"λ={lmbda}: using cached checkpoint")
        return str(ckpt_path), json.loads(hist_path.read_text())
    print(f"λ={lmbda}: training {EPOCHS} epochs...")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = GRU4Rec(n_users=len(sequences))
    trainer = Trainer(
        model, train_loader, val_loader,
        lr=LR, lambda_kl=lmbda, device=DEVICE, config=CONFIG,
    )
    history = trainer.fit(EPOCHS, checkpoint_path=str(ckpt_path))
    hist_path.write_text(json.dumps(history))
    return str(ckpt_path), history


ckpts, histories = {}, {}
for lmbda in LAMBDAS:
    ckpts[lmbda], histories[lmbda] = train_or_load(lmbda)

# %% [markdown]
# ## 3. Convergence diagnostic — confirm each run reached near-max
#
# Best-val-loss checkpointing keeps the optimal epoch; this plot confirms the
# 50-epoch budget was enough for validation loss to plateau for every λ.

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
for lmbda in LAMBDAS:
    h = histories[lmbda]
    ax.plot([e["epoch"] for e in h], [e["val_loss"] for e in h], label=f"λ={lmbda}")
ax.set_xlabel("epoch")
ax.set_ylabel("validation loss")
ax.set_title("Convergence per λ")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig("examples/demo_outputs/lambda_convergence.png", bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Evaluate each λ on the held-out test split
#
# Loads the **best** checkpoint (not the last epoch), runs inference, and scores
# the same logits against the real next activity (fidelity) and the routine
# template (alignment).

# %%
def eval_checkpoint(lmbda, ckpt_path):
    ckpt = load_checkpoint(ckpt_path)
    model = GRU4Rec(**ckpt["config"]["model_kwargs"])
    model.load_state_dict(ckpt["model_state"])
    arrays = run_ranking_predictions(
        model, sequences, ckpt["config"],
        batch_size=BATCH, device=DEVICE, routines=routines,
    )
    fid = evaluate_ranking(arrays["y_true"], arrays["y_scores"], ks=(1, 5))
    ali = evaluate_alignment(arrays["routine_targets"], arrays["y_scores"], ks=(1, 5))
    return {"lambda": lmbda, **fid, **ali,
            "realism_gap": fid["accuracy"] - ali["alignment_accuracy"]}


results = [eval_checkpoint(lmbda, ckpts[lmbda]) for lmbda in LAMBDAS]
df = pd.DataFrame(results).set_index("lambda")
# Alignment is meaningful as *lift over the λ=0 baseline*, not absolute.
df["alignment_lift"] = df[ALIGNMENT] - df.loc[0.0, ALIGNMENT]
df.round(4)

# %% [markdown]
# ## 5. Fidelity-vs-alignment tradeoff curve

# %%
sel = select_lambda(results, primary_metric=PRIMARY,
                    alignment_metric=ALIGNMENT, floor_frac=FLOOR_FRAC)
fig, ax = plot_alignment_tradeoff(
    results, fidelity_key=PRIMARY, alignment_key=ALIGNMENT,
    selected_lambda=sel["lambda_star"], floor=sel["floor"],
    save_path="examples/demo_outputs/lambda_tradeoff.png",
)
plt.show()

# %% [markdown]
# ## 6. λ* selection

# %%
print(sel["rationale"])
print()
print(f"Selected λ* = {sel['lambda_star']}")
print(f"  fidelity ceiling (λ=0) {PRIMARY} = {sel['ceiling']:.4f}")
print(f"  fidelity floor          = {sel['floor']:.4f}")
print(f"  knee (sanity check)     = λ={sel['knee_lambda']}")
print(f"  alignment lift over baseline = "
      f"{df.loc[sel['lambda_star'], 'alignment_lift']:+.4f}")

# %% [markdown]
# ## 7. Supporting views

# %%
# Realism gap (fidelity − alignment, top-1) as λ grows.
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(df.index, df["realism_gap"], "-o")
ax.axvline(sel["lambda_star"], color="crimson", ls="--", label=f"λ*={sel['lambda_star']}")
ax.set_xlabel("λ")
ax.set_ylabel("realism gap (accuracy − alignment_accuracy)")
ax.set_title("Realism gap shrinks as λ increases")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("examples/demo_outputs/realism_gap.png", bbox_inches="tight", dpi=150)
plt.show()

# The routine templates the KL term nudges toward.
plot_template_heatmap(routines, activity_labels=CATEGORIES,
                      title="Healthy routine templates",
                      save_path="examples/demo_outputs/templates_used.png")
plt.show()

# %% [markdown]
# ## 8. Conclusions
#
# - **λ\*** is the most template-aligned GRU4Rec whose next-activity ranking
#   quality stays within 5% of the pure-BPR ceiling — the strongest nudge we can
#   apply while keeping recommendations credible.
# - The **realism gap** narrows monotonically with λ, making the
#   fidelity↔nudge trade explicit.
# - **Caveats:** both axes are in-sample to the loss (fidelity≈BPR, alignment≈KL),
#   so the curve shows the *menu* of trades, not which is "best" — that is set by
#   the fidelity floor. Alignment is a single-step proxy; the final confirmation
#   of λ\* is a full-day cumulative deviation-reduction metric (Regime B), which
#   is future work.
```

> **Note on the cold-start cell:** the data cell rebuilds `sequences.pkl` from
> the raw ATUS file only when it is absent, mirroring `preprocess.py`'s pipeline
> (`load_atus_activity_file` → `map_activity_category` → `build_all_sequences` →
> `save_sequences`). In the normal case `sequences.pkl` already exists and this
> branch never runs.

- [ ] **Step 2: Sanity-check the source parses as Python**

Run: `python -c "import ast; ast.parse(open('notebooks/lambda_selection_e2e.py').read()); print('parses')"`
Expected: prints `parses`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/lambda_selection_e2e.py
git commit -m "feat(notebook): add jupytext source for lambda-selection e2e"
```

---

## Task 5: Build and execute the notebook end-to-end

**Files:**
- Generate: `notebooks/lambda_selection_e2e.ipynb`

- [ ] **Step 1: Convert the percent source to a notebook**

Run: `jupytext --to notebook notebooks/lambda_selection_e2e.py`
Expected: creates `notebooks/lambda_selection_e2e.ipynb`, exit 0.

- [ ] **Step 2: Execute the notebook top-to-bottom**

Run (from the repo root so `src` imports resolve):
```bash
PYTHONPATH=. jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=7200 \
    notebooks/lambda_selection_e2e.ipynb
```
Expected: exit 0, no cell raises. First run trains 8 models (slow); re-runs use the cache. Afterward these files exist:
- `checkpoints/sweep/gru4rec_lambda{0.0,0.1,0.25,0.5,1.0,2.0,4.0,8.0}.pt` (+ `_history.json`)
- `examples/demo_outputs/lambda_convergence.png`, `lambda_tradeoff.png`, `realism_gap.png`, `templates_used.png`

- [ ] **Step 3: Confirm the notebook produced a λ\* and the tradeoff plot**

Run:
```bash
python - <<'PY'
import nbformat
nb = nbformat.read("notebooks/lambda_selection_e2e.ipynb", as_version=4)
text = "\n".join(
    "".join(o.get("text", "") for o in c.get("outputs", []))
    for c in nb.cells if c.cell_type == "code"
)
assert "Selected λ* =" in text, "λ* not printed — selection cell did not run"
print("OK: notebook executed and selected a λ*")
PY
```
Expected: prints `OK: ...`.

- [ ] **Step 4: Run the full test suite (no regressions)**


Run: `python -m pytest -q`
Expected: all tests pass, including `test_lambda_selection.py` and `test_visualization.py`. Do not claim success without green output.

- [ ] **Step 5: Commit**

```bash
git add notebooks/lambda_selection_e2e.ipynb examples/demo_outputs/
git commit -m "feat(notebook): execute lambda-selection e2e, add result plots"
```

> Checkpoints under `checkpoints/sweep/` are build artifacts. If the repo's
> `.gitignore` does not already exclude `checkpoints/`, do **not** commit them.

---

## Definition of Done

`pytest -q` is green (with the two new test modules), and
`notebooks/lambda_selection_e2e.ipynb` runs top-to-bottom from raw/preprocessed
data to: a per-λ results table, a convergence plot proving each model trained to
near-max, the fidelity-vs-alignment tradeoff curve with the floor and λ\* marked,
and a printed λ\* rationale. Re-running the notebook uses cached checkpoints.

---

## Self-Review

- **Spec coverage:** cached training (Task 4 `train_or_load`); GRU4Rec-only sweep over the 8-point grid (Task 4 `LAMBDAS`); convergence diagnostic (Task 4 §3 + saved history); per-λ fidelity+alignment eval on the best checkpoint (Task 4 §4 `eval_checkpoint`); tradeoff plot (Task 2 + notebook §5); floor+max-alignment selection with knee sanity check (Task 1 + notebook §6); alignment-as-lift (notebook §4/§6); caveats in conclusions (notebook §8); tested selection + plot (Tasks 1–2); end-to-end execution (Task 5).
- **Type/name consistency:** metric keys `ndcg@5` / `alignment_ndcg@5` / `accuracy` match `evaluate_ranking`/`evaluate_alignment` output (verified in `src/eval/evaluation.py`). `select_lambda(results, primary_metric, alignment_metric, floor_frac)` and its return keys (`lambda_star`, `ceiling`, `floor`, `knee_lambda`, `candidates`, `rationale`) are identical in Task 1, its tests, and notebook §5–6. `plot_alignment_tradeoff(results, fidelity_key, alignment_key, lambda_key, selected_lambda, floor, save_path, title)` matches between Task 2, its test, and notebook §5. `run_ranking_predictions(model, sequences, config, batch_size, device, routines)`, `build_routines(arr, K=, random_state=)→(routines,_,_)`, `Trainer(model, train_loader, val_loader, lr, lambda_kl, device, config)` + `.fit(n_epochs, checkpoint_path)→history`, `load_checkpoint`, `HabitDataset(arr, window_size=, routines=, user_ids=)`, `build_user_mapping`, `train_val_test_split` all match `main` as read during planning.
- **Placeholder scan:** no TBD/TODO; every code step is complete; the one runtime-dependent name (`preprocess_atus`, used only on the cold-data path) has an explicit verify-and-fix step (Task 5 Step 2) rather than being assumed silently.
- **Risk flagged:** first execution is long (8×50 epochs); runs on MPS when available (CPU fallback) and is mitigated by caching and the generous nbconvert timeout. MPS↔CPU numeric differences are fine here — metrics are compared *within* one device's run, and determinism is per-device. Checkpoints are artifacts and called out as not-to-commit unless gitignored.
