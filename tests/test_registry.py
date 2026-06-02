# tests/test_registry.py
import pytest

from src.models.registry import get_model_class
from src.models.gru4rec import GRU4Rec


def test_get_model_class_gru4rec():
    assert get_model_class("gru4rec") is GRU4Rec


def test_unknown_model_raises_value_error():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_class("nope")


def test_unimplemented_ablation_raises_module_not_found():
    # LSTMRec / TransformerRec are planned but not yet implemented.
    with pytest.raises(ModuleNotFoundError, match="not yet implemented"):
        get_model_class("lstm")
    with pytest.raises(ModuleNotFoundError, match="not yet implemented"):
        get_model_class("transformer")
