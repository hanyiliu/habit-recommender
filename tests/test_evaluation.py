import torch
import pytest

from src.eval.evaluation import hit_at_k, mrr


def test_hit_at_1_correct():
    logits  = torch.tensor([[0.1, 0.9, 0.3]])
    targets = torch.tensor([1])
    assert hit_at_k(logits, targets, k=1) == pytest.approx(1.0)


def test_hit_at_1_wrong():
    logits  = torch.tensor([[0.1, 0.9, 0.3]])
    targets = torch.tensor([0])
    assert hit_at_k(logits, targets, k=1) == pytest.approx(0.0)


def test_hit_at_k_in_top5():
    # 5 classes, correct is last in descending ranking
    logits  = torch.tensor([[0.5, 0.4, 0.3, 0.2, 0.1]])
    targets = torch.tensor([4])
    assert hit_at_k(logits, targets, k=5) == pytest.approx(1.0)
    assert hit_at_k(logits, targets, k=1) == pytest.approx(0.0)


def test_hit_at_k_batch_average():
    # 2 examples: first correct, second wrong → average = 0.5
    logits  = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    targets = torch.tensor([0, 0])
    assert hit_at_k(logits, targets, k=1) == pytest.approx(0.5)


def test_mrr_rank_1():
    logits  = torch.tensor([[0.9, 0.1, 0.5, 0.3]])
    targets = torch.tensor([0])     # highest score → rank 1
    assert mrr(logits, targets) == pytest.approx(1.0)


def test_mrr_rank_2():
    logits  = torch.tensor([[0.9, 0.1, 0.5, 0.3]])
    targets = torch.tensor([2])     # 2nd-highest → rank 2
    assert mrr(logits, targets) == pytest.approx(0.5)


def test_mrr_batch_average():
    # rank-1 (MRR=1.0) and rank-2 (MRR=0.5) → average = 0.75
    logits  = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    targets = torch.tensor([0, 0])  # first correct (rank 1), second wrong (rank 2)
    assert mrr(logits, targets) == pytest.approx(0.75)
