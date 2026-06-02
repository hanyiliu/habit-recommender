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
