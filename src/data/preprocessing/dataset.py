import numpy as np
import torch
from torch.utils.data import Dataset

from src.models.utils.routine_matcher import RoutineMatcher


class HabitDataset(Dataset):
    """
    Sliding-window dataset over daily activity sequences.

    Each sample yields:
        x        (window_size,) long tensor — input activity slots
        y        scalar long tensor         — next activity category
        user_id  scalar long tensor         — respondent index

    When ``routines`` is supplied a fourth element is appended:
        routine_target  scalar long tensor  — activity at the predicted slot
                         taken from the nearest routine template (the target of
                         the alignment cross-entropy term in ``combined_loss``).

    Args:
        sequences    (N, num_slots) int array of category indices per slot
        window_size  number of observed slots used as input context;
                     must be in [1, num_slots - 1]
        routines     optional (K, num_slots) int array of routine templates.
                     When given, each sample also returns a routine target.
        user_ids     optional (N,) int array mapping each row to a global user
                     index. Defaults to the row index, which is the right
                     choice when ``sequences`` already holds every user; pass
                     explicit ids when ``sequences`` is a split so embeddings
                     stay consistent across train/val/test.

    Windows are generated lazily in __getitem__ — sequences are stored once
    as a single tensor rather than materializing every window up front.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        window_size: int = 12,
        routines: np.ndarray = None,
        user_ids: np.ndarray = None,
    ):
        sequences = np.asarray(sequences)
        if sequences.ndim != 2:
            raise ValueError(
                f"sequences must be 2-D (N, num_slots), got shape {sequences.shape}"
            )
        if window_size <= 0 or window_size >= sequences.shape[1]:
            raise ValueError(
                f"window_size must be in [1, {sequences.shape[1] - 1}], got {window_size}"
            )

        self.sequences_np = sequences.astype(np.int64)
        self.sequences = torch.from_numpy(self.sequences_np)
        self.window_size = window_size
        self.windows_per_seq = sequences.shape[1] - window_size

        if user_ids is None:
            self.user_ids = np.arange(sequences.shape[0], dtype=np.int64)
        else:
            user_ids = np.asarray(user_ids, dtype=np.int64)
            if user_ids.shape != (sequences.shape[0],):
                raise ValueError(
                    f"user_ids must have shape ({sequences.shape[0]},), "
                    f"got {user_ids.shape}"
                )
            self.user_ids = user_ids

        self.matcher = RoutineMatcher(routines) if routines is not None else None

    def __len__(self) -> int:
        return len(self.sequences) * self.windows_per_seq

    def __getitem__(self, idx: int) -> tuple:
        row = idx // self.windows_per_seq
        t = idx % self.windows_per_seq
        seq = self.sequences[row]
        x = seq[t: t + self.window_size]
        y = seq[t + self.window_size]
        user_id = torch.tensor(int(self.user_ids[row]), dtype=torch.long)

        if self.matcher is None:
            return x, y, user_id

        slot_t = t + self.window_size
        rt = int(self.matcher.get_targets(self.sequences_np[row][np.newaxis, :], slot_t)[0])
        return x, y, user_id, torch.tensor(rt, dtype=torch.long)


def build_user_mapping(sequences: dict) -> dict:
    """Map each respondent id to a 0-indexed integer, sorted for determinism."""
    return {uid: idx for idx, uid in enumerate(sorted(sequences.keys()))}


def train_val_test_split(
    sequences: dict,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> tuple:
    """Split a {respondent_id -> sequence} dict by user into train/val/test dicts."""
    rng = np.random.default_rng(seed)
    keys = np.array(list(sequences.keys()))
    rng.shuffle(keys)
    n = len(keys)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_keys = keys[:n_test]
    val_keys = keys[n_test: n_test + n_val]
    train_keys = keys[n_test + n_val:]
    return (
        {k: sequences[k] for k in train_keys},
        {k: sequences[k] for k in val_keys},
        {k: sequences[k] for k in test_keys},
    )
