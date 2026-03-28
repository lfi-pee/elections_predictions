from __future__ import annotations
import hashlib

import torch
import torch.nn as nn

from src.data_token import DataToken


class StringEmbedding(nn.Module):
    def __init__(self, num_buckets: int, dim: int) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.embedding = nn.Embedding(num_buckets, dim)

    def forward(self, strings: list[str]) -> torch.Tensor:
        device = self.embedding.weight.device
        hashes = [
            int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % self.num_buckets
            for s in strings
        ]
        indices = torch.tensor(hashes, dtype=torch.long, device=device)
        return self.embedding(indices)


class TokenEmbedding(nn.Module):
    def __init__(self, d_model: int = 128, num_buckets: int = 10000) -> None:
        super().__init__()
        self.d_identity = d_model - 32

        self.date_proj = nn.Linear(1, self.d_identity)
        self.value_proj = nn.Linear(1, 32)

        self.election_emb = StringEmbedding(num_buckets, self.d_identity)
        self.location_emb = StringEmbedding(num_buckets, self.d_identity)
        self.candidate_emb = StringEmbedding(num_buckets, self.d_identity)
        self.party_emb = StringEmbedding(num_buckets, self.d_identity)
        self.metric_emb = StringEmbedding(num_buckets, self.d_identity)

        self.mask_token = nn.Parameter(torch.randn(32))

    def _embed_identity(self, flat_tokens: list[DataToken]) -> torch.Tensor:
        device = self.date_proj.weight.device
        date_tensor = torch.tensor(
            [[t.date_float] for t in flat_tokens], dtype=torch.float32, device=device
        )
        identity = self.date_proj(date_tensor)
        identity += self.election_emb([t.election_type for t in flat_tokens])
        identity += self.location_emb([t.location for t in flat_tokens])
        identity += self.candidate_emb([t.candidate for t in flat_tokens])
        identity += self.party_emb([t.party for t in flat_tokens])
        identity += self.metric_emb([t.metric_type for t in flat_tokens])
        return identity

    def _embed_value(
        self, flat_tokens: list[DataToken], flat_masked: list[bool]
    ) -> torch.Tensor:
        device = self.value_proj.weight.device

        val_inputs = [
            [0.0] if m else [t.value] for t, m in zip(flat_tokens, flat_masked)
        ]
        val_tensor = torch.tensor(val_inputs, dtype=torch.float32, device=device)
        val_embedded = self.value_proj(val_tensor)

        mask_tensor = torch.tensor(
            flat_masked, dtype=torch.bool, device=device
        ).unsqueeze(1)
        val_embedded = torch.where(
            mask_tensor, self.mask_token.expand_as(val_embedded), val_embedded
        )
        return val_embedded

    def forward(
        self, tokens: list[list[DataToken]], masked_indices: list[list[bool]]
    ) -> torch.Tensor:
        batch_size = len(tokens)
        seq_len = len(tokens[0])

        flat_tokens = [t for seq in tokens for t in seq]
        flat_masked = [m for seq in masked_indices for m in seq]

        identity = self._embed_identity(flat_tokens)
        val_embedded = self._embed_value(flat_tokens, flat_masked)

        combined = torch.cat([identity, val_embedded], dim=1)
        return combined.view(batch_size, seq_len, -1)


class UniversalMaskedSetTransformer(nn.Module):
    """
    Universal Masked Set Transformer (UMST)
    Takes an arbitrary set of DataTokens, some of which have their values masked.
    Predicts the missing values using global set self-attention.
    """

    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 4) -> None:
        super().__init__()
        self.token_embedding = TokenEmbedding(d_model=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.value_head = nn.Linear(d_model, 100)

    def forward(
        self,
        tokens: list[list[DataToken]],
        masked_indices: list[list[bool]],
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.token_embedding(tokens, masked_indices)
        out = self.transformer(x, src_key_padding_mask=padding_mask)
        return self.value_head(out)
