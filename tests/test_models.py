import torch

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
