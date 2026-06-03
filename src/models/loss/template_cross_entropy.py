"""Cross-entropy of the model's prediction against the routine-template target (the nudge term)."""
import torch
import torch.nn.functional as F


def template_cross_entropy(
    logits: torch.Tensor,
    routine_targets: torch.Tensor,
) -> torch.Tensor:
    """Softmax cross-entropy against the nearest routine template's activity.

    This is the alignment / nudging term: it pushes the model's distribution
    toward the activity the matched healthy template prescribes at the predicted
    slot. It is the same loss function as the fidelity term — plain
    cross-entropy — just pointed at a different target. (With a one-hot target it
    also equals KL(one_hot || softmax(logits)), since the target's entropy is 0.)

    Args:
        logits (torch.Tensor, (B, n_activities)): Unnormalized scores
        routine_targets (torch.Tensor, (B,)): Activity index from the nearest routine at slot T

    Returns:
        torch.Tensor: Scalar cross-entropy loss against the template target.
    """
    return F.cross_entropy(logits, routine_targets)
