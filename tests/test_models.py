import pytest
import torch

from src.models.gru4rec import GRU4Rec
from src.models.lstm_rec import LSTMRec
from src.models.transformer_rec import TransformerRec


def _inputs(n_users=5, n_activities=11, B=4, T=24):
    sequences = torch.randint(0, n_activities, (B, T))
    user_ids = torch.randint(0, n_users, (B,))
    return sequences, user_ids


def test_lstm_rec_forward_shape():
    model = LSTMRec(n_users=5)
    sequences, user_ids = _inputs(n_users=5)
    logits = model(sequences, user_ids)
    assert logits.shape == (4, 11)


def test_transformer_rec_forward_shape():
    model = TransformerRec(n_users=5)
    sequences, user_ids = _inputs(n_users=5)
    logits = model(sequences, user_ids)
    assert logits.shape == (4, 11)


def test_models_respect_n_activities():
    seqs, uids = _inputs(n_users=3, n_activities=7)
    assert LSTMRec(n_users=3, n_activities=7)(seqs, uids).shape == (4, 7)
    assert TransformerRec(n_users=3, n_activities=7)(seqs, uids).shape == (4, 7)


@pytest.mark.parametrize("ModelClass", [GRU4Rec, LSTMRec, TransformerRec])
def test_use_user_embedding_false_ignores_user_ids(ModelClass):
    # With the user embedding off, the model must (a) still produce (B, 11)
    # logits, (b) have no user_embed parameter, and (c) be completely invariant
    # to user_ids — the same sequence yields the same output for any user.
    torch.manual_seed(0)
    model = ModelClass(n_users=5, use_user_embedding=False).eval()
    sequences = torch.randint(0, 11, (4, 24))
    out_a = model(sequences, torch.zeros(4, dtype=torch.long))
    out_b = model(sequences, torch.full((4,), 3, dtype=torch.long))
    assert out_a.shape == (4, 11)
    assert not hasattr(model, "user_embed")
    assert torch.allclose(out_a, out_b)


@pytest.mark.parametrize("ModelClass", [GRU4Rec, LSTMRec, TransformerRec])
def test_use_user_embedding_true_uses_user_ids(ModelClass):
    # Default keeps the user embedding and user_ids do affect the output.
    torch.manual_seed(0)
    model = ModelClass(n_users=5).eval()
    assert hasattr(model, "user_embed")
    sequences = torch.randint(0, 11, (4, 24))
    out_a = model(sequences, torch.zeros(4, dtype=torch.long))
    out_b = model(sequences, torch.full((4,), 3, dtype=torch.long))
    assert not torch.allclose(out_a, out_b)
