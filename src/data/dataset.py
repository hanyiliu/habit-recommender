# src/data/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.models.utils.routine_matcher import RoutineMatcher


class ATUSDataset(Dataset):
    """Sliding-window next-slot prediction dataset.

    For each user sequence (48 slots) and each target slot t in
    [window_size, 47], produces one example:
      - context:        (window_size,) LongTensor — seq[t-window_size : t]
      - user_idx:       scalar LongTensor
      - target:         scalar LongTensor — seq[t]
      - routine_target: scalar LongTensor — nearest-routine activity at slot t
    """

    def __init__(
        self,
        sequences: dict,       # TUCASEID -> (48,) int8 array
        user_to_idx: dict,     # TUCASEID -> int
        routines: np.ndarray,  # (K, 48) int array
        window_size: int = 24,
    ):
        matcher = RoutineMatcher(routines)
        items = []
        for tucaseid, seq in sequences.items():
            uid = user_to_idx[tucaseid]
            full_seq = seq.astype(np.int64)
            full_seq_batch = full_seq[np.newaxis, :]          # (1, 48)
            for t in range(window_size, 48):
                context = full_seq[t - window_size : t].copy()
                target = int(full_seq[t])
                rt = int(matcher.get_targets(full_seq_batch, t)[0])
                items.append((context, uid, target, rt))
        self.items = items
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        context, uid, target, rt = self.items[idx]
        return (
            torch.tensor(context, dtype=torch.long),
            torch.tensor(uid,     dtype=torch.long),
            torch.tensor(target,  dtype=torch.long),
            torch.tensor(rt,      dtype=torch.long),
        )


def build_user_mapping(sequences: dict) -> dict:
    """Map each TUCASEID to a 0-indexed integer, sorted for determinism."""
    return {uid: idx for idx, uid in enumerate(sorted(sequences.keys()))}


def train_val_test_split(
    sequences: dict,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[dict, dict, dict]:
    """Split sequences by user into train / val / test dicts."""
    rng = np.random.default_rng(seed)
    keys = np.array(list(sequences.keys()))
    rng.shuffle(keys)
    n = len(keys)
    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)
    test_keys  = keys[:n_test]
    val_keys   = keys[n_test : n_test + n_val]
    train_keys = keys[n_test + n_val :]
    return (
        {k: sequences[k] for k in train_keys},
        {k: sequences[k] for k in val_keys},
        {k: sequences[k] for k in test_keys},
    )
