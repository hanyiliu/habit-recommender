"""Combined cross-entropy + KL loss for routine-aware next-activity recommendation."""
import torch
import torch.nn.functional as F

from src.models.loss.kl_loss import kl_loss


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    routine_targets: torch.Tensor,
    lambda_kl: float = 0.5,
) -> torch.Tensor:
    """L = CrossEntropy(logits, targets) + lambda_kl * L_KL

    The fidelity term is plain softmax cross-entropy against the true next
    activity. With only 11 activity classes the model already scores every
    candidate, so full softmax cross-entropy is the natural, lower-variance
    choice (no negative sampling needed).

    Args:
        logits (torch.Tensor, (B, n_activities)): Unnormalized scores
        targets (torch.Tensor, (B,)): True next activity indices
        routine_targets (torch.Tensor, (B,)): Activity from nearest routine template at slot T
        lambda_kl (float, optional): Weight of KL term. 0.0 = fidelity-only ablation. Defaults to 0.5.

    Returns:
        torch.Tensor: Scalar combined loss.
    """
    return F.cross_entropy(logits, targets) + lambda_kl * kl_loss(logits, routine_targets)
