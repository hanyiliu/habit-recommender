"""Bayesian Personalised Ranking loss for next-activity recommendation."""
import torch
import torch.nn.functional as F


def bpr_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_neg: int = 10,
) -> torch.Tensor:
    """Implements BPR loss, which ranks true next activity above n_neg sampled negatives.

    Args:
        logits (torch.Tensor, (B, n_activities)): Unnormalized scores
        targets (torch.Tensor, (B,)): True next activity indices
        n_neg (int, optional): Number of negative samples per positive. Defaults to 10.

    Returns:
        torch.Tensor: Scalar mean BPR loss.
    """
    B, n_activities = logits.shape
    pos_scores = logits[torch.arange(B, device=logits.device), targets]  # (B,)

    neg = torch.randint(0, n_activities, (B, n_neg), device=logits.device)
    conflict = neg == targets.unsqueeze(1)
    neg[conflict] = (neg[conflict] + 1) % n_activities
    row_idx = torch.arange(B, device=logits.device).unsqueeze(1).expand(B, n_neg)
    neg_scores = logits[row_idx, neg]                             # (B, n_neg)
    pos_expanded = pos_scores.unsqueeze(1).expand_as(neg_scores)  # (B, n_neg)
    return -F.logsigmoid(pos_expanded - neg_scores).mean()
