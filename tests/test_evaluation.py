import numpy as np
import pytest

from src.eval.evaluation import hit_rate_at_k, next_activity_accuracy


def test_hit_rate_at_1_correct():
    y_scores = np.array([[0.1, 0.9, 0.3]])
    y_true = np.array([1])
    assert hit_rate_at_k(y_true, y_scores, k=1) == pytest.approx(1.0)


def test_hit_rate_at_1_wrong():
    y_scores = np.array([[0.1, 0.9, 0.3]])
    y_true = np.array([0])
    assert hit_rate_at_k(y_true, y_scores, k=1) == pytest.approx(0.0)


def test_hit_rate_at_k_in_top5():
    y_scores = np.array([[0.5, 0.4, 0.3, 0.2, 0.1]])
    y_true = np.array([4])
    assert hit_rate_at_k(y_true, y_scores, k=5) == pytest.approx(1.0)
    assert hit_rate_at_k(y_true, y_scores, k=1) == pytest.approx(0.0)


def test_accuracy_batch_average():
    y_scores = np.array([[0.9, 0.1], [0.1, 0.9]])
    y_true = np.array([0, 0])  # first correct, second wrong → 0.5
    assert next_activity_accuracy(y_true, y_scores) == pytest.approx(0.5)


def test_evaluate_model_smoke():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from train_main import evaluate_model

    class DummyModel(nn.Module):
        def forward(self, context, user_ids):
            return torch.zeros(context.size(0), 11)

    n = 8
    context         = torch.randint(0, 11, (n, 4))
    targets         = torch.randint(0, 11, (n,))
    user_ids        = torch.randint(0, 3,  (n,))
    routine_targets = torch.randint(0, 11, (n,))
    loader = DataLoader(TensorDataset(context, targets, user_ids, routine_targets), batch_size=4)

    metrics = evaluate_model(DummyModel(), loader)
    assert "accuracy" in metrics
    assert "hit_rate@5" in metrics
    assert all(isinstance(v, float) for k, v in metrics.items() if not k.startswith("per_class"))
