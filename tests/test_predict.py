# tests/test_predict.py
import numpy as np
import pytest
import torch

from src.models.gru4rec import GRU4Rec
from src.eval.predict.runner import (
    run_ranking_predictions, save_predictions, load_checkpoint,
    predict_from_checkpoint,
)


def _fake_sequences(n: int = 20, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {i: rng.integers(0, 11, size=48, dtype=np.int8) for i in range(n)}


def _config(n_users: int, window: int = 24) -> dict:
    return {"model": "gru4rec", "model_kwargs": {"n_users": n_users},
            "window": window, "val_frac": 0.15, "test_frac": 0.15,
            "seed": 42, "k_routines": 10, "n_classes": 11}


def test_run_ranking_predictions_shapes_and_ranges():
    seqs = _fake_sequences(20)
    cfg = _config(len(seqs), window=24)
    model = GRU4Rec(n_users=len(seqs))
    out = run_ranking_predictions(model, seqs, cfg, batch_size=16)

    assert set(out) == {"y_true", "y_scores", "time_slots", "user_ids"}
    n = out["y_true"].shape[0]
    # 3 test users (20 * 0.15) x (48 - 24) windows = 72
    assert n == 3 * (48 - 24)
    assert out["y_scores"].shape == (n, 11)
    assert out["time_slots"].min() >= 24 and out["time_slots"].max() <= 47
    assert out["user_ids"].min() >= 0 and out["user_ids"].max() < len(seqs)


def test_run_ranking_predictions_deterministic():
    seqs = _fake_sequences(20)
    cfg = _config(len(seqs))
    model = GRU4Rec(n_users=len(seqs))
    a = run_ranking_predictions(model, seqs, cfg, batch_size=16)
    b = run_ranking_predictions(model, seqs, cfg, batch_size=16)
    assert np.array_equal(a["y_true"], b["y_true"])
    assert np.array_equal(a["time_slots"], b["time_slots"])
    assert np.array_equal(a["user_ids"], b["user_ids"])
    assert np.array_equal(a["y_scores"], b["y_scores"])


def test_save_predictions_roundtrip(tmp_path):
    arrays = {"y_true": np.zeros(4, np.int64),
              "y_scores": np.zeros((4, 11), np.float32),
              "time_slots": np.arange(4, dtype=np.int64),
              "user_ids": np.arange(4, dtype=np.int64)}
    out = tmp_path / "p.npz"
    save_predictions(arrays, str(out), "gru4rec")
    with np.load(out) as d:
        assert str(d["model"]) == "gru4rec"
        assert d["y_scores"].shape == (4, 11)


def test_load_checkpoint_requires_config(tmp_path):
    ckpt = tmp_path / "bad.pt"
    torch.save({"model_state": {}}, ckpt)  # no 'config'
    with pytest.raises(ValueError, match="no 'config'"):
        load_checkpoint(str(ckpt))


def test_predict_from_checkpoint_end_to_end(tmp_path):
    seqs = _fake_sequences(20)
    import pickle
    seq_path = tmp_path / "sequences.pkl"
    seq_path.write_bytes(pickle.dumps(seqs))

    model = GRU4Rec(n_users=len(seqs))
    ckpt = tmp_path / "best.pt"
    torch.save({"model_state": model.state_dict(),
                "config": _config(len(seqs))}, ckpt)

    out = tmp_path / "predictions_gru4rec.npz"
    predict_from_checkpoint(str(ckpt), str(seq_path), str(out))
    with np.load(out) as d:
        assert {"y_true", "y_scores", "time_slots", "user_ids"} <= set(d.files)


def test_run_ranking_predictions_emits_routine_targets():
    seqs = _fake_sequences(20)
    cfg = _config(len(seqs), window=24)
    model = GRU4Rec(n_users=len(seqs))
    rng = np.random.default_rng(1)
    routines = rng.integers(0, 11, size=(3, 48), dtype=np.int64)  # hand-made templates
    out = run_ranking_predictions(
        model, seqs, cfg, batch_size=16, routines=routines,
    )
    assert "routine_targets" in out
    n = out["y_true"].shape[0]
    assert out["routine_targets"].shape == (n,)
    assert out["routine_targets"].min() >= 0
    assert out["routine_targets"].max() < 11


def test_run_ranking_predictions_no_routines_keeps_four_keys():
    seqs = _fake_sequences(20)
    cfg = _config(len(seqs), window=24)
    model = GRU4Rec(n_users=len(seqs))
    out = run_ranking_predictions(model, seqs, cfg, batch_size=16)
    assert "routine_targets" not in out
    assert set(out) == {"y_true", "y_scores", "time_slots", "user_ids"}
