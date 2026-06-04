#!/usr/bin/env python3
"""Train a habit-recommender model.

Usage:
    python train_main.py                                  # GRU4Rec, defaults
    python train_main.py --model lstm --epochs 50
    python train_main.py --lambda-kl 0.0                 # BPR-only ablation
    python train_main.py --device cuda
"""
from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.preprocessing.dataset import (
    HabitDataset,
    sequences_dict_to_array,
    user_split,
)
from src.models.gru4rec import GRU4Rec
from src.scoring.scoring import build_routines
from src.training.train import Trainer


def _load_model(name: str, n_users: int) -> torch.nn.Module:
    if name == "gru4rec":
        return GRU4Rec(n_users=n_users)
    if name == "lstm":
        try:
            from src.models.lstm_rec import LSTMRec
            return LSTMRec(n_users=n_users)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "LSTMRec is not yet implemented. "
                "See docs/superpowers/plans/2026-05-26-ablation-models.md."
            ) from None
    if name == "transformer":
        try:
            from src.models.transformer_rec import TransformerRec
            return TransformerRec(n_users=n_users)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "TransformerRec is not yet implemented. "
                "See docs/superpowers/plans/2026-05-26-ablation-models.md."
            ) from None
    raise ValueError(f"Unknown model: {name!r}")


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train habit-recommender model")
    p.add_argument("--sequences",  default="data/processed/sequences.pkl",
                   help="Path to preprocessed sequences pickle (output of preprocess.py)")
    p.add_argument("--model",      choices=["gru4rec", "lstm", "transformer"],
                   default="gru4rec")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--lambda-kl",  type=float, default=0.5,
                   help="KL loss weight; 0.0 = BPR-only ablation")
    p.add_argument("--k-routines", type=int,   default=10,
                   help="Number of K-means clusters for build_routines")
    p.add_argument("--window",     type=int,   default=12,
                   help="Sliding-window context size (slots)")
    p.add_argument("--val-frac",   type=float, default=0.15)
    p.add_argument("--test-frac",  type=float, default=0.15)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--predictions", default="data/processed/predictions.npz",
                   help="Where to save test-set predictions for evaluate.py")
    p.add_argument("--device",     default="cpu")
    return p.parse_args()


def main():
    args = build_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    seq_path = Path(args.sequences)
    if not seq_path.exists():
        raise FileNotFoundError(
            f"Sequences file not found: {seq_path}\n"
            "Run 'python preprocess.py' first."
        )
    with open(seq_path, "rb") as f:
        seq_dict = pickle.load(f)

    # Convert {TUCASEID: (48,) array} -> (N, 48) array
    sequences, tucaseids = sequences_dict_to_array(seq_dict)
    n_users = len(sequences)
    print(f"Loaded {n_users} respondents")

    train_idx, val_idx, test_idx = user_split(
        n_users, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test users")

    # Build routine templates from training users only
    train_seqs = sequences[train_idx]
    routines, _, _ = build_routines(train_seqs, K=args.k_routines, random_state=args.seed)
    print(f"Built {len(routines)} routine templates")

    # HabitDataset uses global user indices so embeddings are consistent across splits
    train_ds = HabitDataset(sequences[train_idx], window_size=args.window, routines=routines)
    val_ds   = HabitDataset(sequences[val_idx],   window_size=args.window, routines=routines)
    test_ds  = HabitDataset(sequences[test_idx],  window_size=args.window, routines=routines)
    print(f"Examples: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = _load_model(args.model, n_users=n_users)
    print(f"Model: {type(model).__name__} | params: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(
        model, train_loader, val_loader,
        lr=args.lr,
        lambda_kl=args.lambda_kl,
        device=args.device,
    )
    trainer.fit(args.epochs, checkpoint_path=args.checkpoint)

    # Save test-set predictions in the format expected by evaluate.py
    y_true, y_scores = trainer.predict(test_loader)
    pred_path = Path(args.predictions)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(pred_path, y_true=y_true, y_scores=y_scores)
    print(f"Saved predictions to {pred_path}")
    print(f"Run 'python evaluate.py --predictions {pred_path}' for full metrics.")


if __name__ == "__main__":
    main()
