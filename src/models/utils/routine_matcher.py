"""Helper class for finding closest optimal routine.
"""
import numpy as np

class RoutineMatcher:
    """Helper class for finding closest optimal routine.
    """
    def __init__(self, routines: np.ndarray):
        if routines.ndim != 2 or routines.shape[1] != 48:
            raise ValueError(f"routines must be of shape (K, 48), got {routines.shape}")
        self.routines = routines.astype(np.float32)

    def get_targets(self, sequences: np.ndarray, slot_t: int) -> np.ndarray:
        """Find target optimal routine for given sequence, and the optimal next routine.

        Args:
            sequences (np.ndarray, (B, 48)): slot buffer, where each item
                is one daily routine, total of B items in one batch.
            slot_t (int in [1, 47]): Slot index to be predicted.

        Raises:
            ValueError: Raised when slot_t is outside of expected range.

        Returns:
            np.ndarray (B,): Matching activity at slot_t from the nearest template.
        """
        if sequences.ndim != 2 or sequences.shape[1] != 48:
            raise ValueError(f"sequences must be of shape (B, 48), got {sequences.shape}")
        if slot_t < 1 or slot_t >= 48:
            raise ValueError(f"slot_t must be in [1, 47], got {slot_t}")
        partial_seqs = sequences[:, :slot_t] # (B, slot_t)
        partial_routines = self.routines[:, :slot_t] #(K, slot_t)

        # Hamming distance: count mismatching slots (activity IDs are categorical, not ordinal)
        distances = (partial_seqs[:, np.newaxis, :] != partial_routines[np.newaxis, :, :]).sum(axis=-1)  # (B, K)

        best_idx = distances.argmin(axis=1) # (B,)

        return self.routines[best_idx, slot_t].astype(np.int64)
