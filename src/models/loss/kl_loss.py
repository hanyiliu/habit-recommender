"""KL divergence loss between model output and routine template target."""
import torch
import torch.nn.functional as F


def kl_loss(
    logits: torch.Tensor,
    routine_targets: torch.Tensor,
) -> torch.Tensor:
    """KL divergence between model output and routine template one-hot target.

    Implemented as cross-entropy, equivalent to KL(one_hot || softmax(logits)).

    Args:
        logits (torch.Tensor, (B, n_activities)): Unnormalized scores
        routine_targets (torch.Tensor, (B,)): Activity index from the nearest routine at slot T

    Returns:
        torch.Tensor: Scalar KL loss.
    """
    return F.cross_entropy(logits, routine_targets)
