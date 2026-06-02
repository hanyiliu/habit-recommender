# src/models/registry.py
"""Single source of truth for mapping a model name to its class."""
from src.models.gru4rec import GRU4Rec


def get_model_class(name: str):
    if name == "gru4rec":
        return GRU4Rec
    if name == "lstm":
        try:
            from src.models.lstm_rec import LSTMRec
            return LSTMRec
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "LSTMRec is not yet implemented. "
                "See docs/superpowers/plans/2026-05-26-ablation-models.md."
            ) from None
    if name == "transformer":
        try:
            from src.models.transformer_rec import TransformerRec
            return TransformerRec
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "TransformerRec is not yet implemented. "
                "See docs/superpowers/plans/2026-05-26-ablation-models.md."
            ) from None
    raise ValueError(f"Unknown model: {name}")
