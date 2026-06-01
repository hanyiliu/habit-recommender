# tests/test_dataset.py
import numpy as np
import torch

from src.data.preprocessing.dataset import (
    HabitDataset,
    build_user_mapping,
    train_val_test_split,
)


def _fake_sequences(n: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {i: rng.integers(0, 11, size=48, dtype=np.int8) for i in range(n)}


def _fake_routines(k: int = 2) -> np.ndarray:
    return np.zeros((k, 48), dtype=np.int64)


def test_dataset_length():
    arr = np.stack([_fake_sequences(3)[i] for i in range(3)])
    ds = HabitDataset(arr, window_size=24)
    # 3 users × (48 - 24) windows each
    assert len(ds) == 3 * (48 - 24)


def test_dataset_item_shapes_with_routines():
    arr = np.stack([_fake_sequences(1)[0]])
    ds = HabitDataset(arr, window_size=8, routines=_fake_routines())
    context, target, uid, rt = ds[0]
    assert context.shape == (8,)
    assert target.shape == ()
    assert uid.shape == ()
    assert rt.shape == ()


def test_dataset_three_tuple_without_routines():
    arr = np.stack([_fake_sequences(1)[0]])
    ds = HabitDataset(arr, window_size=8)
    item = ds[0]
    assert len(item) == 3


def test_dataset_values_in_range():
    arr = np.stack([_fake_sequences(2)[i] for i in range(2)])
    ds = HabitDataset(arr, window_size=8, routines=_fake_routines())
    context, target, uid, rt = ds[0]
    assert 0 <= target.item() <= 10
    assert 0 <= rt.item() <= 10
    assert context.min() >= 0
    assert context.max() <= 10


def test_dataset_dtypes():
    arr = np.stack([_fake_sequences(1)[0]])
    ds = HabitDataset(arr, window_size=4, routines=_fake_routines())
    context, target, uid, rt = ds[0]
    assert context.dtype == torch.long
    assert target.dtype == torch.long
    assert uid.dtype == torch.long
    assert rt.dtype == torch.long


def test_dataset_user_ids_global():
    arr = np.stack([_fake_sequences(2)[i] for i in range(2)])
    ds = HabitDataset(arr, window_size=8, user_ids=np.array([5, 9]))
    _, _, uid0 = ds[0]
    _, _, uid1 = ds[ds.windows_per_seq]
    assert uid0.item() == 5
    assert uid1.item() == 9


def test_build_user_mapping_contiguous():
    seqs = {100: None, 200: None, 50: None}
    u2i = build_user_mapping(seqs)
    assert set(u2i.values()) == {0, 1, 2}
    assert len(u2i) == 3


def test_train_val_test_split_partition():
    seqs = _fake_sequences(100)
    train, val, test = train_val_test_split(seqs, val_frac=0.15, test_frac=0.15, seed=0)
    assert len(train) + len(val) + len(test) == 100
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)


def test_train_val_test_split_deterministic():
    seqs = _fake_sequences(100)
    t1, v1, te1 = train_val_test_split(seqs, seed=42)
    t2, v2, te2 = train_val_test_split(seqs, seed=42)
    assert set(t1) == set(t2)
    assert set(v1) == set(v2)
