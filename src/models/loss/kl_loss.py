import torch
import torch.nn.functional as F

def kl_loss(
    logits: torch.FloatTensor,
    routine_targets: torch.LongTensor,
) -> torch.FloatTensor:
    """KL divergence between model output and routine template one-hot target.

    Implemented as cross-entropy, equivalent to KL(one_hot || softmax(logits)).

    Args:
        logits:          (B, n_activities) unnormalized scores
        routine_targets: (B,) activity index from the nearest routine at slot T

    Returns:
        Scalar KL loss.
    """
    return F.cross_entropy(logits, routine_targets)