"""Tiny offline fixtures for engineering checks, never research results."""

from types import SimpleNamespace

import torch
from torch import nn


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 32, width: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.q_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.head = nn.Linear(width, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        mask = attention_mask.unsqueeze(-1)
        context = (x * mask).cumsum(1) / mask.cumsum(1).clamp_min(1)
        hidden = torch.tanh(self.q_proj(context)) * torch.sigmoid(self.v_proj(context))
        return SimpleNamespace(logits=self.head(self.dropout(hidden)))


def toy_records(count: int = 12, seed: int = 10) -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    records = []
    for i in range(count):
        tokens = torch.randint(1, 32, (6 + i % 4,), generator=generator).tolist()
        records.append({'id': str(i), 'skill': 'A', 'group': str(i // 2), 'split': 'train',
                        'input_ids': tokens, 'labels': [-100] * 3 + tokens[3:]})
    return records
