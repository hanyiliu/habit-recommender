import torch
import torch.nn as nn


class TransformerRec(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_activities: int = 11,
        activity_dim: int = 64,
        user_dim: int = 64,
        hidden_size: int = 128,
        n_layers: int = 1,
    ):
        super().__init__()
        self.activity_embed = nn.Embedding(n_activities, activity_dim)
        self.user_embed = nn.Embedding(n_users, user_dim)
        self.input_proj = nn.Linear(activity_dim + user_dim, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(hidden_size, n_activities)

    def forward(
        self,
        sequences: torch.LongTensor,
        user_ids: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Forward pass for the TransformerRec model.

        Args:
            sequences (torch.LongTensor, (B, T)): Activity indices for slots 0..T-1
            user_ids (torch.LongTensor, (B,)): User index per sample

        Returns:
            logits (torch.FloatTensor, (B, n_activities)): Predictions of next activity
        """
        B, T = sequences.shape
        act_emb = self.activity_embed(sequences)
        user_emb = self.user_embed(user_ids).unsqueeze(1).expand(-1, T, -1)
        x = self.input_proj(torch.cat([act_emb, user_emb], dim=-1))  # (B, T, hidden)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=sequences.device)
        x = self.transformer(x, mask=mask, is_causal=True)           # (B, T, hidden)
        logits = self.output_proj(x[:, -1, :])                        # (B, n_activities)
        return logits
