import torch
import torch.nn as nn


class LSTMRec(nn.Module):
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
        self.lstm = nn.LSTM(
            input_size=activity_dim + user_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_size, n_activities)

    def forward(
        self,
        sequences: torch.LongTensor,
        user_ids: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Forward pass for the LSTMRec model.

        Args:
            sequences (torch.LongTensor, (B, T)): Activity indices for slots 0..T-1
            user_ids (torch.LongTensor, (B,)): User index per sample

        Returns:
            logits (torch.FloatTensor, (B, n_activities)): Predictions of next activity
        """
        B, T = sequences.shape
        act_emb = self.activity_embed(sequences)
        user_emb = self.user_embed(user_ids).unsqueeze(1).expand(-1, T, -1)
        lstm_input = torch.cat([act_emb, user_emb], dim=-1)    # (B, T, 128)
        output, _ = self.lstm(lstm_input)                       # (B, T, hidden)
        logits = self.output_proj(output[:, -1, :])             # (B, n_activities)
        return logits
