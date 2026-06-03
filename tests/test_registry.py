# tests/test_registry.py
import pytest

from src.models.registry import get_model_class
from src.models.gru4rec import GRU4Rec
from src.models.lstm_rec import LSTMRec
from src.models.transformer_rec import TransformerRec


def test_get_model_class_gru4rec():
    assert get_model_class("gru4rec") is GRU4Rec


def test_get_model_class_lstm():
    assert get_model_class("lstm") is LSTMRec


def test_get_model_class_transformer():
    assert get_model_class("transformer") is TransformerRec


def test_unknown_model_raises_value_error():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_class("nope")
