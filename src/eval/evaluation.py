import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def hit_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Fraction of examples where the true label appears in the top-k predictions."""
    k = min(k, logits.size(-1))
    top_k = torch.topk(logits, k, dim=-1).indices      # (B, k)
    hits  = (top_k == targets.unsqueeze(-1)).any(-1)   # (B,)
    return hits.float().mean().item()


def mrr(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean reciprocal rank of the true label."""
    sorted_idx = torch.argsort(logits, dim=-1, descending=True)  # (B, C)
    # (row, col) pairs where col is the rank (0-based) of the target
    rank_positions = (sorted_idx == targets.unsqueeze(-1)).nonzero(as_tuple=False)
    ranks = rank_positions[:, 1].float() + 1  # 1-based rank
    return (1.0 / ranks).mean().item()


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: str = "cpu") -> dict:
    """Return hit@1, hit@5, and MRR over a DataLoader."""
    model.eval()
    all_logits, all_targets = [], []
    for context, user_ids, targets, _ in loader:
        logits = model(context.to(device), user_ids.to(device)).cpu()
        all_logits.append(logits)
        all_targets.append(targets)
    if not all_logits:
        return {"hit@1": 0.0, "hit@5": 0.0, "mrr": 0.0}
    logits  = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    return {
        "hit@1": hit_at_k(logits, targets, k=1),
        "hit@5": hit_at_k(logits, targets, k=5),
        "mrr":   mrr(logits, targets),
    }
