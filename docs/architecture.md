# Habit Recommender — Architecture & Implementation Status

This is the canonical description of the system as it is actually built. Where
an earlier spec draft and the implementation disagreed, this document follows
the **code** when the code's choice is the correct one (those points are called
out inline), and tracks not-yet-implemented items in
[§7 Implementation Status](#7-implementation-status).

---

## 1. Dataset: American Time Use Survey (ATUS)

The ATUS activity files (`atusact_XXXX.dat`) record each respondent's day as
timestamped activities with start time, stop time, duration, and two-level
activity codes (Tier-1 broad category, Tier-2 specific).

### Data representation
- Each respondent's day is converted into a fixed-length sequence of **48 time
  slots** (30 min each); the ATUS diary day starts at 4:00 AM.
- Each slot is assigned one of **11 consolidated activity categories**. The
  canonical index order is defined in `src/utils/activity_map.py::CATEGORIES`:
  `Sleep(0), Grooming(1), Work(2), Education(3), Eating(4), Socializing(5),
  Leisure/Screen(6), Household(7), Exercise(8), Travel(9), Other(10)`.
- Categories are mapped from ATUS Tier-1/Tier-2 codes via a hand-crafted lookup
  table (`activity_map.py`: `TIER1_TO_CATEGORY` + Tier-2 overrides for Personal
  Care and Leisure).
- Each 30-min slot is assigned the **dominant** activity overlapping it; unfilled
  slots default to `Other` (`preprocessor.py::build_sequence`).

### Role in the system
- Population-level sequences are clustered into **routine templates** —
  behavioral archetypes used as recommendation targets.
- Templates represent the *healthiest* version of each behavioral cluster across
  the day.

---

## 2. Routine Template Builder (offline)

Runs once during training setup (`src/scoring/scoring.py`) and produces `K`
optimal daily routine templates used at inference to nudge users toward
healthier behavior.

### Step 1 — Clustering
- k-means over all respondent 48-slot vectors (`build_routines`,
  `KMeans(n_clusters=K, n_init=10)`).
- Sweep `K ∈ {5, 10, 20, 30}` via `run_routine_sweep`.

### Step 2 — Individual health scoring
Each **respondent** is scored individually (not the centroid) with a composite
health score from five hand-crafted features (`scoring.py`):
- **Sleep contiguity** — longest contiguous Sleep run (`score_sleep_contiguity`).
- **Exercise presence** — weighted Exercise-slot count (`score_exercise_presence`).
- **Meal regularity** — proximity of Eating slots to meal-time anchors
  (`score_meal_regularity`).
- **Screen-time penalty** — Leisure/Screen slots in the late-evening window
  (`score_screen_time`).
- **Work-hour structure** — whether Work bleeds into sleep hours
  (`score_work_structure`).

Features are normalized to `[0, 1]` and combined as a weighted sum
(`DEFAULT_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]`), justified by CDC sleep and
WHO exercise guidance.

### Step 3 — Template construction
- Within each cluster, filter to respondents above a score threshold (default
  top **50%**).
- Take the **per-slot mode** of the filtered sequences → one optimal template
  vector per cluster, shape `(48,)` (`_slot_mode`, used by `build_routines`).

  > **Correction vs. earlier spec:** the template is the **per-slot mode**, not
  > the mean. Activity categories are *nominal* labels (Sleep=0 … Other=10), so
  > averaging indices is meaningless (mean of Sleep=0 and Work=2 would be
  > Grooming=1). The mode reflects the most common activity at each slot among
  > high-scorers.

- Clusters where fewer than **30** respondents pass the threshold are merged
  into the nearest valid cluster by centroid distance (`min_cluster_size=30`).
- Threshold is a hyperparameter — ablate top-25% / 50% / 75% (`run_routine_sweep`
  `thresholds=(0.25, 0.50, 0.75)`).

Templates are stored as a `(K, 48)` integer matrix.

---

## 3. Model Architecture: GRU4Rec

Primary model (`src/models/gru4rec.py`): a GRU-based sequential recommender,
personalized via user-ID embeddings and nudged toward healthy archetypes by a
template-alignment loss.

### Input
- The user's observed activity slots so far in the day — a partial sequence of
  length `T < 48` (produced by `HabitDataset`'s sliding window).

### Embedding layer
- Activity category index → dense embedding, `activity_dim = 64`.
- User ID → learned embedding, `user_dim = 64`.
- Concatenated at each step → GRU input size `2d = 128`.

### GRU layers
- 1–2 stacked GRU layers (default 1), hidden size `H = 128`, processed
  left-to-right; final hidden state encodes context up to `T`.

### Template matching (parallel step)
- The partial sequence is matched to the nearest template and the target
  activity for slot `T+1` is retrieved (`RoutineMatcher.get_targets`).
- Matching uses **Hamming distance** (count of mismatching slots) over the
  observed prefix.

  > **Correction vs. earlier spec:** distance is **Hamming**, not Euclidean.
  > Activity IDs are categorical/nominal, so Euclidean distance would impose a
  > false ordinal scale; Hamming counts slot-wise disagreements, which is the
  > correct notion of similarity for label sequences.

### Output layer
- Linear projection `H=128 → 11` logits; softmax gives a distribution over the
  next activity. With only 11 categories, all candidates are scored
  exhaustively — no separate retrieval stage.

### Loss function
`L = L_BPR + λ · L_KL` (`src/models/loss/combined_loss.py`):
- **`L_BPR`** — sample negatives, rank the true next activity above them
  (`bpr_loss`, `n_neg=10`).
- **`L_KL`** — KL divergence between the model's distribution and a one-hot at
  the template's target activity for that slot, implemented as cross-entropy
  (`kl_loss`). The nudging term.
- `λ` (`--lambda-kl`, default 0.5) trades nudging vs. sequence fidelity.

### Comparison models (ablations)
- **LSTM** — same architecture, GRU → LSTM.
- **Transformer** — single-head self-attention over the slot sequence.
- **GRU4Rec (λ=0)** — pure BPR, isolates the template-alignment contribution
  (supported today via `--lambda-kl 0`).

---

## 4. Recommender-systems framing
- **Sequential recommendation** — next-item prediction over a time-ordered
  sequence (GRU4Rec).
- **Personalized ranking** — BPR ranks the correct next activity above
  negatives, per user.
- **Collaborative signals** — the user-ID embedding shares patterns across
  similar users.
- **Template-based recommendation** — matching users to population-derived
  archetypes, analogous to user-based collaborative filtering over clustered
  routine profiles.

---

## 5. Data flow

```
ATUS raw .dat  →  parse + 30-min discretize  →  map to 11 categories
   →  k-means into K archetypes
   →  score individuals → filter top-N per cluster → per-slot mode → K templates
   →  per-user sliding windows + K templates
   →  GRU4Rec training (L_BPR + λ·L_KL)
   →  inference: match partial day → nearest template (Hamming);
      GRU predicts next activity nudged toward template
   →  recommended next activity
```

---

## 6. Pipeline (commands)

```bash
python preprocess.py                                   # raw .dat → data/processed/sequences.pkl
python train_main.py --epochs 5                        # train GRU4Rec, save checkpoints/best.pt
python -m src.eval.predict.gru4rec --checkpoint checkpoints/best.pt   # → predictions_<model>.npz
python evaluate.py --predictions data/processed/predictions_gru4rec.npz
```

- **Split:** respondents are split **80 / 10 / 10** train/val/test
  (`train_main.py` defaults `--val-frac 0.10 --test-frac 0.10`;
  `train_val_test_split` defaults match).
- **Optimizer:** Adam; `ReduceLROnPlateau` on validation loss.

### Evaluation metrics (`src/eval/evaluation.py`)
- **Ranking:** NDCG@K (`ndcg_at_k`), Hit Rate@K (`hit_rate_at_k`),
  next-activity accuracy (`next_activity_accuracy`).
- **Sequence:** Sequence Match Score (`sequence_match_score`).
- **Behavioral:** Routine Similarity (cosine, `routine_similarity_score`
  `mode="frequency"`), Deviation Reduction (`deviation_reduction`).

---

## 7. Implementation Status

What is built vs. what is still planned, so the spec stays honest.

### ✅ Implemented and aligned
- 48-slot / 11-category representation, Tier-1/Tier-2 mapping, dominant-slot
  discretization.
- Health-scored template builder: k-means, 5-feature individual scoring,
  top-fraction filtering, per-slot-mode templates, small-cluster merging,
  K × threshold sweep.
- GRU4Rec (64/64 → 128 → 128 → 11), BPR + λ·KL loss, λ=0 ablation, user-ID
  personalization, Hamming template matching.
- LSTM / Transformer ablation models (`LSTMRec`, `TransformerRec`), sharing
  GRU4Rec's `forward(sequences, user_ids) -> (B, 11)` interface and resolved via
  the shared registry (`--model lstm` / `--model transformer`).
- 80/10/10 split (**aligned in this change**).
- All evaluation metric *definitions* (ranking, sequence, behavioral).
- Train → analysis ranking handoff: checkpoint → `predictions_<model>.npz` →
  `evaluate.py` (ranking + per-class metrics).

### 🟡 Planned — noted, not yet implemented
1. **Model selection by validation NDCG@10.** Current training selects the best
   checkpoint by **validation loss** (combined BPR+KL) and `λ` is not yet tuned
   on NDCG@10 (`src/training/train.py`). Selecting/tuning on validation NDCG@10
   is a planned refinement.
2. **Multi-year ATUS ingestion.** `preprocess.py` currently loads a single year
   (`data/2024_data/atusact_2024.dat`). Concatenating multiple years for
   population-scale coverage is planned.
3. **Sparse-respondent filtering.** Respondents with many unfilled slots are
   currently kept (gaps default to `Other`); a min-filled-slots filter in
   preprocessing is a possible future addition.
4. **Sequence-level evaluation end-to-end (Regime B).** The sequence/behavioral
   metrics exist but are not yet fed by a real **autoregressive full-day
   rollout**, and `evaluate.py`'s sequence path handles a single day rather than
   a population. Design: `docs/superpowers/plans/2026-06-02-train-analysis-bridge.md`
   (Phase B / "Out of scope" section).

### Notes
- Raw data currently lives under `data/2024_data/` (single year), not a generic
  `data/raw/`.
- `routine_similarity_score` defaults to `mode="positional"`; the cosine
  variant the metrics doc refers to is `mode="frequency"`.
