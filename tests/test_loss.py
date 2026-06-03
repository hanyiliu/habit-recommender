import torch
import torch.nn.functional as F

from src.models.loss.combined_loss import combined_loss
from src.models.loss.template_cross_entropy import template_cross_entropy


def _inputs(B=8, n=11, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(B, n)
    targets = torch.randint(0, n, (B,))
    routine = torch.randint(0, n, (B,))
    return logits, targets, routine


def test_combined_loss_is_cross_entropy_plus_lambda_kl():
    # Fidelity term is softmax cross-entropy on the true next activity (no BPR).
    logits, targets, routine = _inputs()
    lam = 0.5
    expected = F.cross_entropy(logits, targets) + lam * template_cross_entropy(logits, routine)
    got = combined_loss(logits, targets, routine, lambda_kl=lam)
    assert torch.allclose(got, expected)


def test_combined_loss_lambda_zero_is_pure_cross_entropy():
    logits, targets, routine = _inputs(seed=1)
    got = combined_loss(logits, targets, routine, lambda_kl=0.0)
    assert torch.allclose(got, F.cross_entropy(logits, targets))


def test_combined_loss_returns_scalar():
    logits, targets, routine = _inputs(seed=2)
    out = combined_loss(logits, targets, routine)
    assert out.dim() == 0
