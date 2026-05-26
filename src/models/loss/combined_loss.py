"""Combined BPR + KL loss for routine-aware next-activity recommendation."""
import torch
from src.models.loss.bpr_loss import bpr_loss
from src.models.loss.kl_loss import kl_loss


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    routine_targets: torch.Tensor,
    lambda_kl: float = 0.5,
) -> torch.Tensor:
    """L = L_BPR + lambda_kl * L_KL

    Args:
        logits (torch.Tensor, (B, n_activities)): Unnormalized scores
        targets (torch.Tensor, (B,)): True next activity indices
        routine_targets (torch.Tensor, (B,)): Activity from nearest routine template at slot T
        lambda_kl (float, optional): Weight of KL term. 0.0 for BPR-only ablation. Defaults to 0.5.

    Returns:
        torch.Tensor: Scalar combined loss.
    """
    return bpr_loss(logits, targets) + lambda_kl * kl_loss(logits, routine_targets)
