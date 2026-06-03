import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.loss.combined_loss import combined_loss
from src.eval.evaluation import evaluate_ranking, evaluate_alignment


class Trainer:
    """Wraps model training, validation, scheduling, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        lambda_align: float = 0.5,
        device: str = "cpu",
        config: dict | None = None,
        track_val_metrics: bool = False,
    ):
        self.model             = model.to(device)
        self.train_loader      = train_loader
        self.val_loader        = val_loader
        self.lambda_align         = lambda_align
        self.device            = device
        self.config            = config
        self.track_val_metrics = track_val_metrics
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
            loss   = combined_loss(logits, targets, routine_targets, self.lambda_align)
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
            loss   = combined_loss(logits, targets, routine_targets, self.lambda_align)
            total += loss.item()
        return total / len(self.val_loader)

    @torch.no_grad()
    def val_ranking_metrics(self) -> dict:
        """Per-epoch validation ranking + alignment metrics.

        Returns accuracy / hit_rate@5 / ndcg@5 measured against both the true
        next activity (fidelity) and the routine-template activity (alignment,
        prefixed ``alignment_``), in a single forward pass over the val set.
        """
        self.model.eval()
        all_logits, all_targets, all_routine = [], [], []
        for context, targets, user_ids, routine_targets in self.val_loader:
            logits = self.model(context.to(self.device), user_ids.to(self.device)).cpu()
            all_logits.append(logits)
            all_targets.append(targets)
            all_routine.append(routine_targets)
        logits  = torch.cat(all_logits).numpy()
        targets = torch.cat(all_targets).numpy()
        routine = torch.cat(all_routine).numpy()
        metrics = evaluate_ranking(targets, logits, ks=(5,))
        metrics.update(evaluate_alignment(routine, logits, ks=(5,)))
        return metrics

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
            record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            if self.track_val_metrics:
                record.update(self.val_ranking_metrics())
            history.append(record)
            if val_loss < self.best_val:
                self.best_val = val_loss
                torch.save(
                    {
                        "epoch":            epoch,
                        "model_state":      self.model.state_dict(),
                        "optimizer_state":  self.optimizer.state_dict(),
                        "scheduler_state":  self.scheduler.state_dict(),
                        "val_loss":         val_loss,
                        "config":           self.config,
                    },
                    checkpoint_path,
                )
            print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")
        return history
