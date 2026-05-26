# Ablation Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement LSTMRec and TransformerRec as ablation variants of GRU4Rec for the habit recommender system.

**Architecture:** Both models share the identical forward signature as GRU4Rec — `forward(sequences: LongTensor (B, T), user_ids: LongTensor (B,)) -> FloatTensor (B, n_activities)`. LSTMRec swaps the GRU cell for an LSTM. TransformerRec replaces the recurrent layer with a single-head causal TransformerEncoder and adds a linear input projection since Transformer d_model must equal hidden_size.

**Tech Stack:** Python 3.10+, PyTorch 2.x

**Naming conventions established in this project:**
- Files live at `src/models/<model_name>.py` (one class per file)
- `RoutineMatcher` in `src/models/utils/routine_matcher.py`
- `GRU4Rec` in `src/models/gru4rec.py`

---

## File Map

| File | Responsibility |
|---|---|
| `src/models/lstm_rec.py` | `LSTMRec` — LSTM-based ablation of GRU4Rec |
| `src/models/transformer_rec.py` | `TransformerRec` — single-head causal Transformer ablation |

---

### Task 1: LSTMRec

**Files:**
- Create: `src/models/lstm_rec.py`

- [ ] **Step 1: Implement LSTMRec**

```python
# src/models/lstm_rec.py
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
```

- [ ] **Step 2: Commit**

```bash
git add src/models/lstm_rec.py
git commit -m "feat: implement LSTMRec ablation model"
```

---

### Task 2: TransformerRec

**Files:**
- Create: `src/models/transformer_rec.py`

- [ ] **Step 1: Implement TransformerRec**

```python
# src/models/transformer_rec.py
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
```

- [ ] **Step 2: Commit**

```bash
git add src/models/transformer_rec.py
git commit -m "feat: implement TransformerRec ablation model"
```
