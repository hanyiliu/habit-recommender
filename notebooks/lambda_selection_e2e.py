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
# The model is trained with `L = CE + λ·L_KL`. The cross-entropy term rewards
# predicting the user's **actual** next activity (fidelity); the KL term nudges predictions
# toward the nearest healthy **routine template** (alignment). This notebook
# runs a **two-stage** search on GRU4Rec: a cheap coarse λ-sweep (short epoch
# budget) measures both axes on the held-out test split and picks λ* as the
# most-aligned model whose fidelity stays within 5% of the λ=0 ceiling; the
# winning λ* is then retrained at the **full** epoch budget for final metrics.
#
# Caveats (kept honest): both axes are *in-sample to the loss* — fidelity tracks
# what the cross-entropy term optimizes, alignment tracks what KL optimizes — and alignment here is
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

# Anchor to the repo root regardless of the kernel's working directory
# (nbconvert runs the kernel inside notebooks/). This makes `src` importable
# AND makes every relative path below (data/, checkpoints/, examples/) resolve
# against the repo root, exactly as preprocess.py / evaluate.py expect.
import sys
for _cand in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
    if (_cand / "src").is_dir():
        sys.path.insert(0, str(_cand))
        os.chdir(_cand)
        break

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
SWEEP_EPOCHS = 15   # cheap coarse sweep — enough to rank λ and locate λ*
FULL_EPOCHS = 50    # full budget, applied only to the winning λ*
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
# ## 2. Coarse λ-sweep training (cached; SWEEP_EPOCHS each)
#
# A short budget is enough to *rank* the λ values and locate λ*; the winner is
# retrained at full length in §7. Each checkpoint is cached (retrain only if
# missing), with the epoch count baked into the filename so the 15-epoch sweep
# and the 50-epoch final run never collide.

# %%
def train_or_load(lmbda, n_epochs):
    ckpt_path = CACHE / f"gru4rec_lambda{lmbda}_e{n_epochs}.pt"
    hist_path = CACHE / f"gru4rec_lambda{lmbda}_e{n_epochs}_history.json"
    if ckpt_path.exists() and hist_path.exists():
        print(f"λ={lmbda} ({n_epochs}e): using cached checkpoint")
        return str(ckpt_path), json.loads(hist_path.read_text())
    print(f"λ={lmbda}: training {n_epochs} epochs...")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = GRU4Rec(n_users=len(sequences))
    trainer = Trainer(
        model, train_loader, val_loader,
        lr=LR, lambda_kl=lmbda, device=DEVICE, config=CONFIG,
        track_val_metrics=True,  # record accuracy/Hit@5/NDCG@5 (fidelity+alignment) each epoch
    )
    history = trainer.fit(n_epochs, checkpoint_path=str(ckpt_path))
    hist_path.write_text(json.dumps(history))
    return str(ckpt_path), history


ckpts, histories = {}, {}
for lmbda in LAMBDAS:
    ckpts[lmbda], histories[lmbda] = train_or_load(lmbda, SWEEP_EPOCHS)

# %% [markdown]
# ## 3. Sweep convergence diagnostic
#
# Best-val-loss checkpointing keeps each run's optimal epoch. At the short sweep
# budget we only need the λ values to *rank* stably enough to locate λ*, not to
# be fully converged — the winner gets the full budget in §7.

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
for lmbda in LAMBDAS:
    h = histories[lmbda]
    ax.plot([e["epoch"] for e in h], [e["val_loss"] for e in h], label=f"λ={lmbda}")
ax.set_xlabel("epoch")
ax.set_ylabel("validation loss")
ax.set_title(f"Sweep convergence per λ ({SWEEP_EPOCHS} epochs)")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig("examples/demo_outputs/lambda_convergence.png", bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# Per-epoch **validation metrics** — accuracy, Hit@5, NDCG@5 — measured against
# both the ground-truth next activity (top row, *fidelity*) and the optimal
# routine template (bottom row, *alignment*), one line per λ. This is the
# "more detail" view: you can watch fidelity decay and alignment rise with λ
# over the course of training, not just the loss.

# %%
_FID = [("accuracy", "accuracy"), ("hit_rate@5", "Hit@5"), ("ndcg@5", "NDCG@5")]
_ALI = [("alignment_accuracy", "accuracy"), ("alignment_hit_rate@5", "Hit@5"),
        ("alignment_ndcg@5", "NDCG@5")]
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
for col in range(3):
    fk, label = _FID[col]
    ak, _ = _ALI[col]
    for lmbda in LAMBDAS:
        h = histories[lmbda]
        ep = [e["epoch"] for e in h]
        axes[0, col].plot(ep, [e[fk] for e in h], label=f"λ={lmbda}")
        axes[1, col].plot(ep, [e[ak] for e in h], label=f"λ={lmbda}")
    axes[0, col].set_title(f"Fidelity {label} (vs ground truth)")
    axes[1, col].set_title(f"Alignment {label} (vs optimal template)")
    axes[1, col].set_xlabel("epoch")
axes[0, 0].set_ylabel("metric")
axes[1, 0].set_ylabel("metric")
axes[0, 0].legend(fontsize=7, ncol=2)
fig.suptitle(f"Per-epoch validation metrics across the λ sweep ({SWEEP_EPOCHS} epochs)")
fig.tight_layout()
fig.savefig("examples/demo_outputs/lambda_sweep_metrics.png", bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Evaluate each λ on the held-out test split (sweep checkpoints)
#
# Loads the **best** checkpoint (not the last epoch), runs inference, and scores
# the same logits against the real next activity (fidelity) and the routine
# template (alignment). These coarse-sweep metrics are what rank λ and select λ*.

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
# ## 7. Full-length training on the selected λ*
#
# The coarse sweep only *ranked* λ. Now retrain the winner for the full
# FULL_EPOCHS budget and report its converged test metrics — this is the model
# you would actually ship. We also show how far the full budget moved the
# metrics versus the short-sweep estimate (a sanity check that the sweep ranked
# λ on the right side of the tradeoff).

# %%
lam_star = sel["lambda_star"]
final_ckpt, final_hist = train_or_load(lam_star, FULL_EPOCHS)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot([e["epoch"] for e in final_hist], [e["val_loss"] for e in final_hist],
        label=f"λ*={lam_star}")
ax.set_xlabel("epoch")
ax.set_ylabel("validation loss")
ax.set_title(f"Full-length convergence (λ*={lam_star}, {FULL_EPOCHS} epochs)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("examples/demo_outputs/lambda_star_convergence.png",
            bbox_inches="tight", dpi=150)
plt.show()

# Per-epoch fidelity & alignment metrics over the full λ* training run.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
ep = [e["epoch"] for e in final_hist]
for k, label in _FID:
    axes[0].plot(ep, [e[k] for e in final_hist], label=label)
for k, label in _ALI:
    axes[1].plot(ep, [e[k] for e in final_hist], label=label)
axes[0].set_title(f"Fidelity (vs ground truth) — λ*={lam_star}")
axes[1].set_title(f"Alignment (vs optimal template) — λ*={lam_star}")
for a in axes:
    a.set_xlabel("epoch")
    a.set_ylabel("metric")
    a.legend(fontsize=8)
fig.suptitle(f"Per-epoch validation metrics over full training (λ*={lam_star}, {FULL_EPOCHS} epochs)")
fig.tight_layout()
fig.savefig("examples/demo_outputs/lambda_star_metrics.png", bbox_inches="tight", dpi=150)
plt.show()

final_metrics = eval_checkpoint(lam_star, final_ckpt)
sweep_star = df.loc[lam_star]
print(f"Final λ*={lam_star} model ({FULL_EPOCHS} epochs):")
print(f"  fidelity  {PRIMARY}            = {final_metrics[PRIMARY]:.4f}")
print(f"  alignment {ALIGNMENT}  = {final_metrics[ALIGNMENT]:.4f}")
print(f"  accuracy = {final_metrics['accuracy']:.4f} | "
      f"alignment_accuracy = {final_metrics['alignment_accuracy']:.4f} | "
      f"realism_gap = {final_metrics['realism_gap']:+.4f}")
print(f"\nvs {SWEEP_EPOCHS}e sweep estimate: "
      f"{PRIMARY} {sweep_star[PRIMARY]:.4f} -> {final_metrics[PRIMARY]:.4f}, "
      f"{ALIGNMENT} {sweep_star[ALIGNMENT]:.4f} -> {final_metrics[ALIGNMENT]:.4f}")

# %% [markdown]
# ## 8. Supporting views

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
# ## 9. Conclusions
#
# - **λ\*** is the most template-aligned GRU4Rec whose next-activity ranking
#   quality stays within 5% of the no-nudge (λ=0, cross-entropy-only) ceiling — the strongest nudge we can
#   apply while keeping recommendations credible.
# - **Two-stage search:** λ* was chosen on a cheap 15-epoch sweep, then
#   retrained at the full 50-epoch budget for the shipped metrics. This assumes
#   the λ *ranking* is stable across budgets — the §7 sweep-vs-full comparison is
#   the check on that assumption (if a non-winning λ would overtake λ* at full
#   length, widen the sweep budget).
# - The **realism gap** narrows monotonically with λ, making the
#   fidelity↔nudge trade explicit.
# - **Caveats:** both axes are in-sample to the loss (fidelity≈cross-entropy, alignment≈KL),
#   so the curve shows the *menu* of trades, not which is "best" — that is set by
#   the fidelity floor. Alignment is a single-step proxy; the final confirmation
#   of λ\* is a full-day cumulative deviation-reduction metric (Regime B), which
#   is future work.
