import numpy as np
import torch
from torch.utils.data import Dataset


class HabitDataset(Dataset):
    """
    Sliding-window dataset over daily activity sequences.

    Each sample yields:
        x        (window_size,) long tensor — input activity slots
        y        scalar long tensor         — next activity category
        user_id  scalar long tensor         — respondent index

    Args:
        sequences    (N, num_slots) int array of category indices per slot
        window_size  number of observed slots used as input context;
                     must be in [1, num_slots - 1]

    Windows are generated lazily in __getitem__ — sequences are stored once
    as a single tensor rather than materializing every window up front.
    """

    def __init__(self, sequences: np.ndarray, window_size: int = 12):
        sequences = np.asarray(sequences)
        if sequences.ndim != 2:
            raise ValueError(
                f"sequences must be 2-D (N, num_slots), got shape {sequences.shape}"
            )
        if window_size <= 0 or window_size >= sequences.shape[1]:
            raise ValueError(
                f"window_size must be in [1, {sequences.shape[1] - 1}], got {window_size}"
            )

        self.sequences = torch.from_numpy(sequences.astype(np.int64))
        self.window_size = window_size
        self.windows_per_seq = sequences.shape[1] - window_size

    def __len__(self) -> int:
        return len(self.sequences) * self.windows_per_seq

    def __getitem__(self, idx: int) -> tuple:
        uid = idx // self.windows_per_seq
        t = idx % self.windows_per_seq
        seq = self.sequences[uid]
        x = seq[t: t + self.window_size]
        y = seq[t + self.window_size]
        return x, y, torch.tensor(uid, dtype=torch.long)
