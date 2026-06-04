from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.loss.combined_loss import combined_loss


class Trainer:
    """Train, validate, and checkpoint a GRU4Rec (or compatible) model.

    Expects DataLoaders that yield (x, y, user_id, routine_target) batches,
    as produced by HabitDataset.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        lambda_kl: float = 0.5,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lambda_kl = lambda_kl
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )
        self.best_val = float("inf")

    def _forward_loss(self, batch) -> torch.Tensor:
        x, y, user_ids, routine_targets = batch
        x = x.to(self.device)
        y = y.to(self.device)
        user_ids = user_ids.to(self.device)
        routine_targets = routine_targets.to(self.device)
        logits = self.model(x, user_ids)
        return combined_loss(logits, y, routine_targets, self.lambda_kl)

    def train_epoch(self) -> float:
        self.model.train()
        total = 0.0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            loss = self._forward_loss(batch)
            loss.backward()
            self.optimizer.step()
            total += loss.item()
        return total / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total = 0.0
        for batch in self.val_loader:
            total += self._forward_loss(batch).item()
        return total / len(self.val_loader)

    def fit(
        self,
        n_epochs: int,
        checkpoint_path: str = "checkpoints/best.pt",
    ) -> list[dict]:
        ckpt_dir = os.path.dirname(checkpoint_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        history = []
        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step(val_loss)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if val_loss < self.best_val:
                self.best_val = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "scheduler_state": self.scheduler.state_dict(),
                        "val_loss": val_loss,
                    },
                    checkpoint_path,
                )
            print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")
        return history

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Run inference over a loader; return (y_true, y_scores) as numpy arrays."""
        self.model.eval()
        all_logits, all_targets = [], []
        for x, y, user_ids, _ in loader:
            logits = self.model(x.to(self.device), user_ids.to(self.device)).cpu()
            all_logits.append(logits)
            all_targets.append(y)
        y_scores = torch.cat(all_logits).numpy()
        y_true = torch.cat(all_targets).numpy()
        return y_true, y_scores
