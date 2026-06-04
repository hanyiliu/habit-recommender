from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.models.utils.routine_matcher import RoutineMatcher


class HabitDataset(Dataset):
    """
    Sliding-window dataset over daily activity sequences.

    Each sample yields:
        x               (window_size,) long tensor — input activity slots
        y               scalar long tensor         — next activity category
        user_id         scalar long tensor         — respondent index
        routine_target  scalar long tensor         — activity at slot t from the
                        nearest routine template (zeros when routines=None)

    Args:
        sequences    (N, num_slots) int array of category indices per slot
        window_size  number of observed slots used as input context;
                     must be in [1, num_slots - 1]
        routines     optional (K, num_slots) int array of routine templates;
                     when provided, routine_target is populated via RoutineMatcher

    Windows are generated lazily in __getitem__; routine targets are precomputed
    once at construction time and stored as a (N, windows_per_seq) tensor.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        window_size: int = 12,
        routines: np.ndarray | None = None,
    ):
        sequences = np.asarray(sequences)
        if sequences.ndim != 2:
            raise ValueError(
                f"sequences must be 2-D (N, num_slots), got shape {sequences.shape}"
            )
        num_slots = sequences.shape[1]
        if window_size <= 0 or window_size >= num_slots:
            raise ValueError(
                f"window_size must be in [1, {num_slots - 1}], got {window_size}"
            )

        self.sequences = torch.from_numpy(sequences.astype(np.int64))
        self.window_size = window_size
        self.windows_per_seq = num_slots - window_size

        N = sequences.shape[0]
        if routines is not None:
            matcher = RoutineMatcher(np.asarray(routines))
            seqs_np = sequences.astype(np.int64)
            rt = np.zeros((N, self.windows_per_seq), dtype=np.int64)
            for uid in range(N):
                row = seqs_np[uid : uid + 1]          # (1, num_slots)
                for t in range(self.windows_per_seq):
                    slot_t = t + window_size
                    rt[uid, t] = matcher.get_targets(row, slot_t)[0]
            self.routine_targets = torch.from_numpy(rt)
        else:
            self.routine_targets = torch.zeros(N, self.windows_per_seq, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sequences) * self.windows_per_seq

    def __getitem__(self, idx: int) -> tuple:
        uid = idx // self.windows_per_seq
        t = idx % self.windows_per_seq
        seq = self.sequences[uid]
        x = seq[t : t + self.window_size]
        y = seq[t + self.window_size]
        return x, y, torch.tensor(uid, dtype=torch.long), self.routine_targets[uid, t]


def sequences_dict_to_array(seq_dict: dict) -> tuple[np.ndarray, list]:
    """Convert TUCASEID-keyed dict to a sorted (N, 48) array plus the key list.

    Returns:
        arr   (N, 48) int64 array
        keys  list of TUCASEIDs in the same row order as arr
    """
    keys = sorted(seq_dict.keys())
    arr = np.stack([seq_dict[k].astype(np.int64) for k in keys])
    return arr, keys


def user_split(
    n_users: int,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly partition user indices into train / val / test sets.

    Returns:
        train_idx, val_idx, test_idx — 1-D int arrays of row indices into the
        (N, 48) sequences array produced by sequences_dict_to_array.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_users)
    n_test = int(n_users * test_frac)
    n_val = int(n_users * val_frac)
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return train_idx, val_idx, test_idx
