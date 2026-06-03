# End-to-End λ-Selection Notebook — Design

**Status:** Approved (brainstorming complete) — 2026-06-02

## Problem

The model is trained with a combined loss `L = L_BPR + λ·L_KL`, where `λ`
trades **fidelity** (predicting the user's actual next activity, via BPR) against
**nudging** (pulling predictions toward the nearest healthy routine template, via
the KL term). There is no single correct `λ`; it sits on a tradeoff curve.

We need one reproducible artifact that goes from raw ATUS data all the way to a
**justified choice of `λ`**, with the supporting graphs — and with the parts that
encode the scientific conclusion (the selection rule and the tradeoff plot) under
test rather than buried in un-runnable notebook cells.

All prerequisites are already on `main`:
- Three models (`gru4rec`, `lstm`, `transformer`) via the shared registry.
- Routine-alignment evaluation: `evaluate_alignment` (`src/eval/evaluation.py`),
  `routine_targets` emitted by `run_ranking_predictions`/`predict_from_checkpoint`
  (`src/eval/predict/runner.py`), and the realism gap reported by `evaluate.py`
  and `train_main.py`.
- Plot helpers in `src/analysis/visualization.py` (`plot_model_comparison`,
  `plot_ablation_comparison`, heatmaps, error-by-slot/activity).

## Goal

A single notebook: raw ATUS → λ-swept GRU4Rec training → fidelity & alignment
evaluation → the fidelity-vs-alignment tradeoff curve and a justified `λ*`.

## Scope decisions (locked during brainstorming)

| Decision | Choice |
|---|---|
| Prerequisite eval | Already merged on `main` (PR #12); no eval work in this plan. |
| Training strategy | **Cache checkpoints; retrain only if the checkpoint is missing.** First run trains; re-runs are fast. |
| Experiment scope | **λ-sweep on GRU4Rec only.** No architecture comparison, no full grid. |
| λ grid | `[0, 0.1, 0.25, 0.5, 1, 2, 4, 8]` (8 points; λ=0 is the fidelity ceiling / pure-BPR baseline). |
| Convergence | **Generous fixed budget `EPOCHS=50` + best-val-loss checkpoint** (existing Trainer behavior) + a per-λ val-loss-vs-epoch diagnostic plot to confirm each run plateaued. No Trainer changes / no early stopping. |
| Primary metric | **`ndcg@5`**. |
| Fidelity floor | **5% relative** to the λ=0 ceiling (`floor_frac=0.05`). |
| Logic placement | **Approach A:** notebook is pure orchestration + narrative; the selection rule and the tradeoff plot are extracted into tested `src/` modules. |

## Architecture

```
sequences.pkl ──► 80/10/10 split ──► build_routines(train)
      │
      └─ for λ in GRID:
            checkpoint cached?  ──yes──► load
                   │ no
                   └─ Trainer.fit(GRU4Rec, λ, EPOCHS) ──► checkpoints/sweep/gru4rec_lambda{λ}.pt
            run_ranking_predictions(model, seqs, cfg, routines=routines)
                   ├─ evaluate_ranking(y_true,  y_scores)   → fidelity metrics
                   └─ evaluate_alignment(routine_targets, y_scores) → alignment metrics
      │
      ▼
  results table (pandas: one row per λ)
      ├─ plot_alignment_tradeoff(...)   → fidelity-vs-alignment curve
      └─ select_lambda(...)             → λ*, floor, knee, rationale
```

The cross-λ comparison is done on **test-set fidelity/alignment metrics**, never
on validation loss (loss contains λ and is not comparable across λ). Within a
single λ, best-val-loss checkpoint selection is fine — it picks the best epoch for
that λ's own objective.

## Components / file structure

### New — `src/analysis/lambda_selection.py` (tested)
Pure selection logic over the per-λ results.

```python
def select_lambda(
    results,                      # list of dicts, each: {"lambda": float, <metric>: float, ...}
    primary_metric: str = "ndcg@5",
    alignment_metric: str = "alignment_ndcg@5",
    floor_frac: float = 0.05,
) -> dict:
    """Choose λ* = the highest-alignment λ whose fidelity stays within
    floor_frac (relative) of the λ=0 fidelity ceiling.

    Returns:
      {
        "lambda_star": float,
        "ceiling": float,          # primary_metric at λ=0
        "floor": float,            # (1 - floor_frac) * ceiling
        "knee_lambda": float,      # max-curvature point (normalized chord distance), sanity check
        "rationale": str,          # human-readable explanation
        "candidates": list,        # λs that cleared the floor
      }
    """
```

- Floor = `(1 - floor_frac) * ceiling`, where `ceiling = primary_metric at λ=0`.
- Among λ with `primary_metric >= floor`, pick the one with max `alignment_metric`.
- Tie-break on alignment ties: prefer the **smaller** λ (less nudging for the same
  alignment is more conservative / credible).
- Knee: normalize both axes to [0,1] across the grid, then pick the point with max
  perpendicular distance from the chord joining the two extreme-λ endpoints
  (Kneedle-style). Reported as a sanity check only — **not** used to choose λ*.

### Modify — `src/analysis/visualization.py` (tested)
Add the tradeoff plot (the existing helpers are bar charts, not a curve).

```python
def plot_alignment_tradeoff(
    results,                       # list of dicts with lambda_key/fidelity_key/alignment_key
    fidelity_key: str = "ndcg@5",
    alignment_key: str = "alignment_ndcg@5",
    lambda_key: str = "lambda",
    selected_lambda: float | None = None,
    floor: float | None = None,
    save_path: str | None = None,
    title: str | None = None,
):
    """Fidelity (y) vs alignment (x), one point per λ annotated with its λ value.
    Highlights selected_lambda and draws the fidelity floor as a horizontal line.
    Returns (fig, ax)."""
```

### New — `notebooks/lambda_selection_e2e.ipynb` (orchestration only)
Sections:
1. **Intro (markdown):** the question, fidelity-vs-alignment framing, the selection rule and its caveats.
2. **Config:** the locked constants above (λ grid, `EPOCHS=50`, `seed=42`, `window=24`, `k_routines=10`, `primary_metric="ndcg@5"`, `floor_frac=0.05`, `CACHE_DIR="checkpoints/sweep"`).
3. **Data:** `load_sequences("data/processed/sequences.pkl")`; if missing, build it via the preprocessor. Deterministic 80/10/10 split; `build_routines` from the train split.
4. **Swept training (cached):** per λ, checkpoint `checkpoints/sweep/gru4rec_lambda{λ}.pt`; skip if present, else `Trainer.fit` GRU4Rec at that λ for `EPOCHS`. Retain each `fit()` history.
5. **Convergence diagnostic:** val-loss vs epoch per λ — visual confirmation each run reached near-max before its metrics are trusted.
6. **Evaluate per λ:** rebuild templates from config, `run_ranking_predictions(..., routines=routines)`, then `evaluate_ranking` + `evaluate_alignment`; assemble a pandas results table (λ, accuracy, hit@5, ndcg@5, alignment_*, realism gap).
7. **Tradeoff plot:** `plot_alignment_tradeoff(ndcg@5 vs alignment_ndcg@5)` with floor line + λ* marked.
8. **Selection:** `select_lambda(...)`; print rationale; report alignment as **lift over the λ=0 baseline**; show knee vs floor-chosen λ*.
9. **Supporting views at λ\*:** realism-gap-vs-λ line; template heatmap; score distribution / error-by-time-slot for the λ* model (reusing existing helpers).
10. **Conclusions (markdown):** chosen λ* with justification, and the honest caveats — both plotted axes are in-sample to the loss; this is a single-step proxy; the cumulative Regime-B outcome metric (deviation reduction over a generated day) is future work.

### Tests
- `tests/test_lambda_selection.py` (new): floor math; floor-respecting max-alignment pick; tie-break prefers smaller λ; knee computation; "alignment monotone in λ" sanity case; λ=0-absent guard (raise a clear error).
- `tests/test_visualization.py` (new): smoke-test `plot_alignment_tradeoff` under the `Agg` backend — returns a `(fig, ax)`, handles `selected_lambda`/`floor`, raises on empty input.

## Data flow / error handling
- **Determinism:** single `seed=42` feeds split, routine building, and the config stored in each checkpoint, so templates are rebuilt identically at eval time.
- **Cache:** checkpoint existence is the cache key; deleting `checkpoints/sweep/` forces a full retrain.
- **Routines:** `build_routines` needs enough train users (min_cluster_size=30); the real dataset (~7,669 respondents) clears this comfortably.
- **Missing data:** if `sequences.pkl` is absent, the data cell builds it from the raw ATUS `.dat` via the preprocessor.

## Verification
- Unit: `pytest tests/test_lambda_selection.py tests/test_visualization.py` green.
- End-to-end: execute the notebook top-to-bottom (`jupyter nbconvert --execute`) and confirm it produces the results table, the tradeoff plot, and a non-empty λ* rationale without errors.

## Out of scope (YAGNI)
- LSTM/Transformer comparison and the full model×λ grid.
- Trainer early stopping.
- Regime-B autoregressive rollout / cumulative deviation-reduction outcome metric.
- Model selection by validation NDCG (training still checkpoints by val loss).
