"""Combined cross-entropy + KL loss for routine-aware next-activity recommendation."""
import torch
import torch.nn.functional as F

from src.models.loss.template_cross_entropy import template_cross_entropy


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    routine_targets: torch.Tensor,
    lambda_kl: float = 0.5,
) -> torch.Tensor:
    """L = CrossEntropy(logits, targets) + lambda_kl * template_cross_entropy

    Both terms are softmax cross-entropy; they differ only in their target. The
    fidelity term scores against the true next activity (with only 11 classes
    the model already scores every candidate, so full cross-entropy is the
    natural, lower-variance choice — no negative sampling). The second term is
    cross-entropy against the routine template, weighted by lambda_kl — the
    nudge toward healthy archetypes.

    Args:
        logits (torch.Tensor, (B, n_activities)): Unnormalized scores
        targets (torch.Tensor, (B,)): True next activity indices
        routine_targets (torch.Tensor, (B,)): Activity from nearest routine template at slot T
        lambda_kl (float, optional): Weight of the template term. 0.0 = fidelity-only ablation. Defaults to 0.5.

    Returns:
        torch.Tensor: Scalar combined loss.
    """
    return (
        F.cross_entropy(logits, targets)
        + lambda_kl * template_cross_entropy(logits, routine_targets)
    )
