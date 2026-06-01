import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.loss.combined_loss import combined_loss


class Trainer:
    """Wraps model training, validation, scheduling, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        lambda_kl: float = 0.5,
        device: str = "cpu",
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.lambda_kl    = lambda_kl
        self.device       = device
        self.optimizer    = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )
        self.best_val     = float("inf")

    def train_epoch(self) -> float:
        if len(self.train_loader) == 0:
            raise ValueError("train_loader is empty — check your dataset and split configuration")
        self.model.train()
        total = 0.0
        for context, targets, user_ids, routine_targets in self.train_loader:
            context         = context.to(self.device)
            user_ids        = user_ids.to(self.device)
            targets         = targets.to(self.device)
            routine_targets = routine_targets.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(context, user_ids)
            loss   = combined_loss(logits, targets, routine_targets, self.lambda_kl)
            loss.backward()
            self.optimizer.step()
            total += loss.item()
        return total / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> float:
        if len(self.val_loader) == 0:
            raise ValueError("val_loader is empty — check your dataset and split configuration")
        self.model.eval()
        total = 0.0
        for context, targets, user_ids, routine_targets in self.val_loader:
            context         = context.to(self.device)
            user_ids        = user_ids.to(self.device)
            targets         = targets.to(self.device)
            routine_targets = routine_targets.to(self.device)
            logits = self.model(context, user_ids)
            loss   = combined_loss(logits, targets, routine_targets, self.lambda_kl)
            total += loss.item()
        return total / len(self.val_loader)

    def fit(
        self,
        n_epochs: int,
        checkpoint_path: str = "checkpoints/best.pt",
    ) -> list[dict]:
        ckpt_dir = os.path.dirname(checkpoint_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        history  = []
        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_epoch()
            val_loss   = self.validate()
            self.scheduler.step(val_loss)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if val_loss < self.best_val:
                self.best_val = val_loss
                torch.save(
                    {
                        "epoch":            epoch,
                        "model_state":      self.model.state_dict(),
                        "optimizer_state":  self.optimizer.state_dict(),
                        "scheduler_state":  self.scheduler.state_dict(),
                        "val_loss":         val_loss,
                    },
                    checkpoint_path,
                )
            print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")
        return history
