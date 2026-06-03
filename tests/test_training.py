import os
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.gru4rec import GRU4Rec
from src.training.train import Trainer


def _make_toy_loader(n: int = 16, T: int = 8, n_users: int = 5) -> DataLoader:
    # Tuple order matches HabitDataset: (context, target, user_id, routine_target)
    context         = torch.randint(0, 11, (n, T))
    targets         = torch.randint(0, 11, (n,))
    user_ids        = torch.randint(0, n_users, (n,))
    routine_targets = torch.randint(0, 11, (n,))
    ds = TensorDataset(context, targets, user_ids, routine_targets)
    return DataLoader(ds, batch_size=4)


def test_train_epoch_returns_positive_float():
    model   = GRU4Rec(n_users=5)
    loader  = _make_toy_loader()
    trainer = Trainer(model, loader, loader)
    loss    = trainer.train_epoch()
    assert isinstance(loss, float)
    assert loss > 0


def test_validate_returns_positive_float():
    model   = GRU4Rec(n_users=5)
    loader  = _make_toy_loader()
    trainer = Trainer(model, loader, loader)
    loss    = trainer.validate()
    assert isinstance(loss, float)
    assert loss > 0


def test_fit_returns_history(tmp_path):
    model   = GRU4Rec(n_users=5)
    loader  = _make_toy_loader()
    ckpt    = str(tmp_path / "best.pt")
    trainer = Trainer(model, loader, loader)
    history = trainer.fit(n_epochs=2, checkpoint_path=ckpt)
    assert len(history) == 2
    assert "epoch"      in history[0]
    assert "train_loss" in history[0]
    assert "val_loss"   in history[0]


def test_fit_saves_checkpoint(tmp_path):
    model   = GRU4Rec(n_users=5)
    loader  = _make_toy_loader()
    ckpt    = str(tmp_path / "best.pt")
    trainer = Trainer(model, loader, loader)
    trainer.fit(n_epochs=2, checkpoint_path=ckpt)
    assert os.path.exists(ckpt)
    saved = torch.load(ckpt, weights_only=False)
    assert "model_state"     in saved
    assert "optimizer_state" in saved
    assert "scheduler_state" in saved
    assert "epoch"           in saved
    assert "val_loss"        in saved


def test_fit_with_lambda_kl_zero(tmp_path):
    # fidelity-only ablation: lambda_kl=0.0 must still train without error
    model   = GRU4Rec(n_users=5)
    loader  = _make_toy_loader()
    ckpt    = str(tmp_path / "best.pt")
    trainer = Trainer(model, loader, loader, lambda_kl=0.0)
    history = trainer.fit(n_epochs=1, checkpoint_path=ckpt)
    assert history[0]["train_loss"] > 0


def test_fit_tracks_val_metrics_when_enabled(tmp_path):
    # With track_val_metrics=True, each history entry carries per-epoch
    # validation ranking + alignment metrics (fidelity vs ground truth AND
    # alignment vs routine template).
    model   = GRU4Rec(n_users=5)
    loader  = _make_toy_loader()
    ckpt    = str(tmp_path / "best.pt")
    trainer = Trainer(model, loader, loader, track_val_metrics=True)
    history = trainer.fit(n_epochs=1, checkpoint_path=ckpt)
    h = history[0]
    for key in ("accuracy", "hit_rate@5", "ndcg@5",
                "alignment_accuracy", "alignment_hit_rate@5", "alignment_ndcg@5"):
        assert key in h, f"missing per-epoch metric {key!r}"


def test_fit_without_tracking_has_no_metric_keys(tmp_path):
    # Default is off: history stays loss-only (no extra eval cost).
    model   = GRU4Rec(n_users=5)
    loader  = _make_toy_loader()
    trainer = Trainer(model, loader, loader)
    history = trainer.fit(n_epochs=1, checkpoint_path=str(tmp_path / "best.pt"))
    assert "accuracy" not in history[0]


def test_fit_saves_config(tmp_path):
    model  = GRU4Rec(n_users=5)
    loader = _make_toy_loader()
    ckpt   = str(tmp_path / "best.pt")
    cfg    = {"model": "gru4rec", "model_kwargs": {"n_users": 5}, "window": 8,
              "val_frac": 0.15, "test_frac": 0.15, "seed": 42,
              "k_routines": 10, "n_classes": 11}
    trainer = Trainer(model, loader, loader, config=cfg)
    trainer.fit(n_epochs=1, checkpoint_path=ckpt)
    saved = torch.load(ckpt, weights_only=False)
    assert saved["config"] == cfg
