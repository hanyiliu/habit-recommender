import torch
from src.models.loss.bpr_loss import bpr_loss
from src.models.loss.kl_loss import kl_loss

def combined_loss(
    logits: torch.FloatTensor,
    targets: torch.LongTensor,
    routine_targets: torch.LongTensor,
    lambda_kl: float = 0.5,
) -> torch.FloatTensor:
    """L = L_BPR + lambda_kl * L_KL

    Args:
        logits:          (B, n_activities) unnormalized scores
        targets:         (B,) true next activity indices
        routine_targets: (B,) activity from nearest routine template at slot T
        lambda_kl:       weight of KL term (0.0 for BPR-only ablation)

    Returns:
        Scalar combined loss.
    """
    return bpr_loss(logits, targets) + lambda_kl * kl_loss(logits, routine_targets)