# habit-recommender

A habit recommender system that learns daily routine patterns from the
[American Time Use Survey (ATUS)](https://www.bls.gov/tus/) and suggests next
activities. Each respondent's day is encoded as a sequence of **48 half-hour
slots** (the ATUS diary starts at 4:00 AM), and each slot holds one of **11
activity categories**:

```
Sleep, Grooming, Work, Education, Eating, Socializing,
Leisure/Screen, Household, Exercise, Travel, Other
```

A GRU4Rec sequence model is trained with a combined **BPR + KL** loss to rank
the next activity given a sliding-window context, with routine "templates"
derived from K-means clustering of training-day sequences.

## Setup

```bash
pip install -r requirements.txt
```

PyTorch may need the appropriate index URL for your platform (CPU vs. GPU
wheels), e.g. for CPU-only:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Pipeline

End-to-end runnable flow:

```bash
python preprocess.py
python train_main.py --epochs 5
python -m src.eval.predict.gru4rec --checkpoint checkpoints/best.pt
python evaluate.py --predictions data/processed/predictions_gru4rec.npz
```

### 1. Preprocess the raw ATUS data

```bash
python preprocess.py
```

Reads the raw activity file `data/2024_data/atusact_2024.dat`, maps each row to
one of the 11 activity categories, builds 48-slot daily sequences, and writes
`data/processed/sequences.pkl` — a dict mapping `TUCASEID -> (48,) int array`.

### 2. Train a model

```bash
python train_main.py
```

Trains **GRU4Rec** with the combined BPR + KL loss, saves the best checkpoint
to `checkpoints/best.pt`, and prints test-set ranking metrics
(`accuracy`, `hit_rate@5`, `ndcg@5`).

Common flags (see `python train_main.py --help` for the full list):

```bash
python train_main.py \
    --model gru4rec \
    --epochs 50 \
    --lambda-kl 0.5 \      # KL loss weight (0.0 = BPR-only ablation)
    --window 24 \          # sliding-window context size in slots
    --k-routines 10 \      # K-means clusters for routine building
    --batch-size 256 \
    --lr 1e-3 \
    --device cpu \
    --checkpoint checkpoints/best.pt
```

> `--model lstm` and `--model transformer` are **planned ablations** and are
> not yet implemented (they raise a clear `ModuleNotFoundError`). See
> `docs/superpowers/plans/2026-05-26-ablation-models.md`.

### 3. Predict (run checkpoint over the held-out test split)

```bash
python -m src.eval.predict.gru4rec --checkpoint checkpoints/best.pt
```

Loads the trained checkpoint, runs it over the held-out test split, and writes
`data/processed/predictions_gru4rec.npz` — an array bundle containing
`y_true`, `y_scores`, `time_slots`, and `user_ids` that `evaluate.py` scores.
The checkpoint is self-describing (it carries its model architecture and the
split configuration used during training), so no extra flags need to be
re-specified.

Other available flags (defaults shown):

```bash
python -m src.eval.predict.gru4rec \
    --checkpoint checkpoints/best.pt \
    --sequences  data/processed/sequences.pkl \
    --out        data/processed/predictions_<model>.npz \
    --batch-size 256 \
    --device     cpu
```

### 4. Evaluation / visualization demo (synthetic data)

```bash
PYTHONPATH=. python examples/evaluation_demo.py
```

Runs the ranking and sequence metrics plus the plotting helpers on **synthetic
data** (no trained model or real data required) and writes PNGs to
`examples/demo_outputs/`.

## Known gaps / roadmap

Honest status of what does **not** yet work:

- **LSTM / Transformer ablations:** `--model lstm` and `--model transformer`
  are planned but not yet implemented (they raise a clear
  `ModuleNotFoundError`). See
  `docs/superpowers/plans/2026-05-26-ablation-models.md`.
- **Full-day autoregressive rollout (Regime B):** sequence-level metrics
  (`sequence_match`, `routine_similarity`, `deviation_reduction`) exist in
  `src/eval/evaluation.py` but are not yet fed by a real full-day rollout; a
  proper autoregressive rollout is needed for these sequence-level scores.
- **Requirements pinning:** `requirements.txt` uses conservative lower bounds
  only; exact pins / a lockfile are a follow-up.
- **CI:** no continuous integration configured yet.

## Project layout

```
habit-recommender/
├── preprocess.py              # entry point: raw .dat -> sequences.pkl
├── train_main.py              # entry point: train GRU4Rec, print metrics
├── evaluate.py                # entry point: score predictions_<model>.npz
├── requirements.txt
├── data/
│   ├── 2024_data/             # raw ATUS files (atusact_2024.dat)
│   └── processed/             # generated: sequences.pkl, predictions.npz
├── src/
│   ├── data/                  # ATUS loading, preprocessing, dataset
│   ├── models/                # GRU4Rec, loss functions, model utils
│   ├── training/              # Trainer (training loop)
│   ├── scoring/               # K-means routine building + routine scoring
│   ├── eval/                  # ranking + sequence metrics; predict/ (checkpoint -> predictions.npz)
│   ├── analysis/              # visualization / plotting helpers
│   └── utils/                 # activity category mapping
├── examples/                  # evaluation_demo.py + demo_outputs/
├── docs/                      # plans and design notes
└── tests/                     # pytest suite
```

## Testing

```bash
pytest tests/
```

PyTorch is required for the dataset and training tests.
