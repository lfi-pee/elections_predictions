from __future__ import annotations

import torch
import torch.nn as nn


class StringEmbedding(nn.Module):
    def __init__(self, num_buckets: int, dim: int) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.embedding = nn.Embedding(num_buckets, dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices)


class TokenEmbedding(nn.Module):
    def __init__(self, d_model: int = 128, num_buckets: int = 50000) -> None:
        super().__init__()
        self.d_identity = d_model - 32

        self.date_proj = nn.Linear(1, self.d_identity)
        self.value_proj = nn.Linear(1, 32)

        self.election_emb = StringEmbedding(num_buckets, self.d_identity)
        # location_emb removed: raw (lat, lon) via geo_proj provides geographic
        # signal without memorising rare commune codes
        self.candidate_emb = StringEmbedding(num_buckets, 3)
        nn.init.zeros_(self.candidate_emb.embedding.weight)
        self.party_emb = StringEmbedding(num_buckets, self.d_identity)
        self.metric_emb = StringEmbedding(num_buckets, self.d_identity)

        # Geo-coordinate projection: (lat, lon) → identity space
        self.geo_proj = nn.Linear(2, self.d_identity)

        self.mask_token = nn.Parameter(torch.randn(32))
        self.layer_norm = nn.LayerNorm(d_model)

    def _embed_identity(self, tokens_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        # Dates are relative to anchored election, typically in [-1.0, 0.0]
        date_tensor = tokens_dict["dates"].unsqueeze(-1)
        identity = self.date_proj(date_tensor)
        
        identity += self.election_emb(tokens_dict["election_type"])
        # location embedding removed — using continuous geo coords instead
        cand_emb_3d = self.candidate_emb(tokens_dict["candidate"])
        identity += torch.nn.functional.pad(cand_emb_3d, (0, self.d_identity - 3))
        identity += self.party_emb(tokens_dict["party"])
        identity += self.metric_emb(tokens_dict["metric_type"])

        # Geo: normalize lat/lon centered on France, then project
        lat = (tokens_dict["latitude"].unsqueeze(-1) - 46.5) / 5.0
        lon = (tokens_dict["longitude"].unsqueeze(-1) - 2.5) / 5.0
        geo_tensor = torch.cat([lat, lon], dim=-1)
        identity += self.geo_proj(geo_tensor)

        return identity

    def _embed_value(
        self, tokens_dict: dict[str, torch.Tensor], masked_indices: torch.Tensor
    ) -> torch.Tensor:
        val_tensor = tokens_dict["values"].unsqueeze(-1)
        
        val_tensor_masked = torch.where(
            masked_indices.unsqueeze(-1), 
            torch.zeros_like(val_tensor), 
            val_tensor
        )
        val_embedded = self.value_proj(val_tensor_masked)

        mask_expanded = masked_indices.unsqueeze(-1)
        val_embedded = torch.where(
            mask_expanded, 
            self.mask_token.view(1, 1, -1).expand_as(val_embedded), 
            val_embedded
        )
        return val_embedded

    def forward(
        self, tokens_dict: dict[str, torch.Tensor], masked_indices: torch.Tensor
    ) -> torch.Tensor:
        identity = self._embed_identity(tokens_dict)
        val_embedded = self._embed_value(tokens_dict, masked_indices)

        combined = torch.cat([identity, val_embedded], dim=-1)
        return self.layer_norm(combined)


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
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.value_head = nn.Linear(d_model, 1)

    def forward(
        self,
        tokens_dict: dict[str, torch.Tensor],
        masked_indices: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.token_embedding(tokens_dict, masked_indices)
        out = self.transformer(x, src_key_padding_mask=padding_mask)
        return self.value_head(out)
