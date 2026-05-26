# Discretize time, build sequences
#
# Converts raw ATUS activity DataFrame into per-respondent 48-slot sequences.
#
# ATUS defines a day as 4:00 AM to 4:00 AM (next day) = 1440 minutes.
# We divide that into 48 slots of 30 minutes each.
# Slot 0 = 04:00–04:30, Slot 1 = 04:30–05:00, ..., Slot 47 = 03:30–04:00

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.activity_map import map_activity_category, CATEGORY_TO_IDX, CATEGORIES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DAY_START_HOUR = 4          # ATUS day starts at 4:00 AM
SLOT_DURATION  = 30         # minutes per slot
N_SLOTS        = 48         # 24 hours / 30 min
OTHER_IDX      = CATEGORY_TO_IDX["Other"]


# ---------------------------------------------------------------------------
# Time parsing
# ---------------------------------------------------------------------------

def time_to_minutes(t: str) -> int:
    """
    Convert HH:MM:SS string to minutes since 4:00 AM (ATUS day start).
    Times before 4:00 AM are treated as next-day (e.g. 03:00 = 23 hours in).

    Returns minutes in [0, 1440).
    """
    h, m, _ = t.strip().split(":")
    h, m = int(h), int(m)
    total = h * 60 + m
    day_start = DAY_START_HOUR * 60  # 240 minutes

    if total >= day_start:
        return total - day_start
    else:
        # Past midnight, before 4 AM — add 1440 - day_start offset
        return total + (1440 - day_start)


# ---------------------------------------------------------------------------
# Single-respondent discretization
# ---------------------------------------------------------------------------

def build_sequence(activities: pd.DataFrame) -> np.ndarray:
    """
    Given all activity rows for one respondent (sorted by start time),
    return a (48,) integer array of category indices.

    Each slot gets the category of whichever activity overlaps it most.
    Unfilled slots are assigned OTHER_IDX.
    """
    sequence = np.full(N_SLOTS, OTHER_IDX, dtype=np.int8)

    # Accumulate minutes per category for each slot
    slot_minutes = np.zeros((N_SLOTS, len(CATEGORIES)), dtype=np.float32)

    for _, row in activities.iterrows():
        try:
            start = time_to_minutes(row["TUSTARTTIM"])
            stop  = time_to_minutes(row["TUSTOPTIME"])
        except Exception:
            continue

        # Handle activities that wrap past midnight (stop < start after offset)
        if stop <= start:
            stop += 1440

        cat_idx = CATEGORY_TO_IDX.get(row["CATEGORY"], OTHER_IDX)

        # Find which slots this activity overlaps
        for slot in range(N_SLOTS):
            slot_start = slot * SLOT_DURATION
            slot_end   = slot_start + SLOT_DURATION

            overlap_start = max(start, slot_start)
            overlap_end   = min(stop,  slot_end)

            if overlap_end > overlap_start:
                slot_minutes[slot, cat_idx] += overlap_end - overlap_start

    # Assign each slot its dominant category
    for slot in range(N_SLOTS):
        if slot_minutes[slot].sum() > 0:
            sequence[slot] = int(np.argmax(slot_minutes[slot]))

    return sequence


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_all_sequences(df: pd.DataFrame) -> dict:
    """
    Build 48-slot sequences for all respondents.

    Args:
        df: DataFrame from load_atus_activity_file, with CATEGORY column added

    Returns:
        dict mapping TUCASEID -> (48,) np.ndarray of category indices
    """
    if "CATEGORY" not in df.columns:
        print("Adding CATEGORY column...")
        df = df.copy()
        df["CATEGORY"] = df.apply(map_activity_category, axis=1)

    sequences = {}
    grouped = df.groupby("TUCASEID")
    total = len(grouped)

    for i, (case_id, group) in enumerate(grouped):
        if i % 1000 == 0:
            print(f"  Processing respondent {i}/{total}...")
        sequences[case_id] = build_sequence(group)

    print(f"Done. Built sequences for {len(sequences)} respondents.")
    return sequences


def save_sequences(sequences: dict, output_path: Path):
    """Serialize sequences dict to a .pkl file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(sequences, f)

    print(f"Saved {len(sequences)} sequences to {output_path}")


def load_sequences(path: Path) -> dict:
    """Load sequences dict from a .pkl file."""
    with open(path, "rb") as f:
        return pickle.load(f)