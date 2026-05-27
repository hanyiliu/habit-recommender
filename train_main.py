#!/usr/bin/env python3
"""Train a habit-recommender model.

Usage:
    python train_main.py                                 # GRU4Rec defaults
    python train_main.py --model lstm --epochs 50
    python train_main.py --model transformer --lambda-kl 0.0
    python train_main.py --model gru4rec --device cuda
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ATUSDataset, build_user_mapping, train_val_test_split
from src.eval.evaluation import evaluate_model
from src.models.gru4rec import GRU4Rec
from src.scoring.scoring import build_routines
from src.training.train import Trainer


def _load_model_class(name: str):
    if name == "gru4rec":
        return GRU4Rec
    if name == "lstm":
        from src.models.lstm_rec import LSTMRec
        return LSTMRec
    if name == "transformer":
        from src.models.transformer_rec import TransformerRec
        return TransformerRec
    raise ValueError(f"Unknown model: {name}")


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train habit-recommender model")
    p.add_argument("--sequences",  default="data/processed/sequences.pkl",
                   help="Path to preprocessed sequences pickle")
    p.add_argument("--model",      choices=["gru4rec", "lstm", "transformer"],
                   default="gru4rec")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--lambda-kl",  type=float, default=0.5,
                   help="KL loss weight (0.0 = BPR-only ablation)")
    p.add_argument("--k-routines", type=int,   default=10,
                   help="Number of K-means clusters for routine building")
    p.add_argument("--window",     type=int,   default=24,
                   help="Sliding-window context size (slots)")
    p.add_argument("--val-frac",   type=float, default=0.15)
    p.add_argument("--test-frac",  type=float, default=0.15)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--device",     default="cpu")
    return p.parse_args()


def main():
    args = build_args()

    sequences = pickle.loads(Path(args.sequences).read_bytes())

    train_seqs, val_seqs, test_seqs = train_val_test_split(
        sequences,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    print(f"Split: {len(train_seqs)} train / {len(val_seqs)} val / {len(test_seqs)} test users")

    train_arr = np.stack([train_seqs[uid] for uid in train_seqs]).astype(float)
    routines, _, _ = build_routines(train_arr, K=args.k_routines, random_state=args.seed)
    print(f"Built {len(routines)} routines from training data")

    user_to_idx = build_user_mapping(sequences)

    train_ds = ATUSDataset(train_seqs, user_to_idx, routines, window_size=args.window)
    val_ds   = ATUSDataset(val_seqs,   user_to_idx, routines, window_size=args.window)
    test_ds  = ATUSDataset(test_seqs,  user_to_idx, routines, window_size=args.window)
    print(f"Dataset sizes: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test examples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    ModelClass = _load_model_class(args.model)
    model = ModelClass(n_users=len(sequences))
    print(f"Model: {ModelClass.__name__} | params: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        lr=args.lr,
        lambda_kl=args.lambda_kl,
        device=args.device,
    )
    trainer.fit(args.epochs, checkpoint_path=args.checkpoint)

    metrics = evaluate_model(model, test_loader, device=args.device)
    print(f"\nTest metrics | hit@1: {metrics['hit@1']:.4f} | hit@5: {metrics['hit@5']:.4f} | MRR: {metrics['mrr']:.4f}")


if __name__ == "__main__":
    main()
