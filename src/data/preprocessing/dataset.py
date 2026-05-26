import numpy as np
import torch
from torch.utils.data import Dataset


class HabitDataset(Dataset):
    """
    Sliding-window dataset over daily activity sequences.

    Each sample yields:
        x        (window_size,) long tensor — input activity slots
        y        scalar long tensor         — next activity category
        user_id  scalar long tensor         — respondent index into sequences array

    Args:
        sequences    (N, 48) int array — output of sequences_to_array()
        window_size  number of observed slots used as input context
    """

    def __init__(self, sequences: np.ndarray, window_size: int = 12):
        windows, targets, user_ids = [], [], []
        for uid, seq in enumerate(sequences):
            for t in range(len(seq) - window_size):
                windows.append(seq[t: t + window_size])
                targets.append(seq[t + window_size])
                user_ids.append(uid)

        self.windows = torch.tensor(np.array(windows), dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple:
        return self.windows[idx], self.targets[idx], self.user_ids[idx]
