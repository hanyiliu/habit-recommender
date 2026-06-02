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
