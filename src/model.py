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
        self.d_model = d_model
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


class LearnableRouter(nn.Module):
    """Top-K token router using identity-only embeddings.
    
    Scores each candidate context token against a pooled target-anchor query,
    selects the top_k most relevant tokens, and multiplies their embeddings
    by normalized relevance scores to create a differentiable gradient path.
    
    Masked (target) tokens are always force-included in the output.
    """

    def __init__(self, d_model: int, d_router: int = 64, top_k: int = 32) -> None:
        super().__init__()
        d_identity = d_model - 32  # match TokenEmbedding's identity dim
        self.d_router = d_router
        self.top_k = top_k

        # Projections for scoring: identity embeddings → router space
        self.query_proj = nn.Linear(d_identity, d_router)
        self.key_proj = nn.Linear(d_identity, d_router)

    def forward(
        self,
        identity_embeddings: torch.Tensor,
        masked_indices: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens based on identity similarity to target tokens.
        
        Args:
            identity_embeddings: (B, N, d_identity) — identity-only embeddings for all candidate tokens
            masked_indices: (B, N) — True for target tokens to predict
            padding_mask: (B, N) — True for padding positions
            
        Returns:
            selected_indices: (B, K) — indices of selected tokens in the original sequence
            scores: (B, K) — normalized relevance scores for selected tokens
            router_stats: (B, N) — raw scores for all tokens (for logging)
        """
        B, N, D = identity_embeddings.shape
        device = identity_embeddings.device

        # Step B: Build target anchor query by pooling masked token embeddings
        # (B, N, d_router)
        keys = self.key_proj(identity_embeddings)
        queries_all = self.query_proj(identity_embeddings)

        # Pool masked token embeddings into anchor query: (B, d_router)
        mask_expanded = masked_indices.unsqueeze(-1).float()  # (B, N, 1)
        mask_count = mask_expanded.sum(dim=1).clamp(min=1.0)  # (B, 1)
        anchor_query = (queries_all * mask_expanded).sum(dim=1) / mask_count  # (B, d_router)

        # Step C: Score all tokens via dot-product
        # (B, N) = (B, 1, d_router) @ (B, d_router, N) → (B, 1, N) → (B, N)
        raw_scores = torch.bmm(anchor_query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)
        raw_scores = raw_scores / (self.d_router ** 0.5)

        # Mask out padding positions with -inf
        if padding_mask is not None:
            raw_scores = raw_scores.masked_fill(padding_mask, float("-inf"))

        # Force-include masked (target) tokens by giving them +inf score
        # so they always appear in top-k
        force_scores = raw_scores.clone()
        force_scores = force_scores.masked_fill(masked_indices, float("inf"))

        # Step D: Top-K selection
        # K = min(top_k, number of non-padding tokens)
        if padding_mask is not None:
            valid_counts = (~padding_mask).sum(dim=1)  # (B,)
            actual_k = min(self.top_k, int(valid_counts.min().item()))
        else:
            actual_k = min(self.top_k, N)
        
        # Ensure we have at least as many slots as masked tokens
        n_masked_max = int(masked_indices.sum(dim=1).max().item())
        actual_k = max(actual_k, n_masked_max)
        actual_k = min(actual_k, N)

        _, topk_indices = torch.topk(force_scores, actual_k, dim=1)  # (B, K)

        # Step E: Compute normalized scores for gradient flow
        # Gather the raw (non-forced) scores for the selected tokens
        selected_raw_scores = torch.gather(raw_scores, 1, topk_indices)  # (B, K)
        
        # For masked positions in the selection, use a neutral score (0 pre-softmax)
        selected_masked = torch.gather(masked_indices, 1, topk_indices)  # (B, K)
        selected_raw_scores = selected_raw_scores.masked_fill(selected_masked, 0.0)
        
        # Softmax over context tokens for gradient weighting  
        normalized_scores = torch.softmax(selected_raw_scores, dim=1)  # (B, K)
        
        # Masked tokens get score 1.0 (unweighted pass-through)
        normalized_scores = torch.where(selected_masked, torch.ones_like(normalized_scores), normalized_scores)

        return topk_indices, normalized_scores, raw_scores


class UniversalMaskedSetTransformer(nn.Module):
    """
    Universal Masked Set Transformer (UMST) with Learnable Token Router.
    
    Takes an arbitrary set of DataTokens (candidate pool), routes them through
    a learned scorer to select the top-K most relevant context tokens, then
    predicts the missing values using global set self-attention on the selected subset.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        d_router: int = 64,
        top_k: int = 32,
        router_warmup_steps: int = 500,
    ) -> None:
        super().__init__()
        self.token_embedding = TokenEmbedding(d_model=d_model)
        self.router = LearnableRouter(d_model=d_model, d_router=d_router, top_k=top_k)
        self.router_warmup_steps = router_warmup_steps
        self._global_step = 0

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

    @property
    def is_router_warming_up(self) -> bool:
        return self._global_step < self.router_warmup_steps

    def _gather_tokens_dict(
        self,
        tokens_dict: dict[str, torch.Tensor],
        indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Gather a subset of tokens_dict using indices (B, K)."""
        gathered = {}
        for k, v in tokens_dict.items():
            if v.dim() == 2:  # (B, N)
                gathered[k] = torch.gather(v, 1, indices)
            else:
                gathered[k] = v
        return gathered

    def forward(
        self,
        tokens_dict: dict[str, torch.Tensor],
        masked_indices: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass with routing.
        
        Returns:
            predictions: (B, K, 1) — value predictions for selected tokens
            route_info: dict with:
                - 'selected_indices': (B, K) indices into original sequence
                - 'selected_masked': (B, K) mask for the selected subset
                - 'selected_tokens': gathered tokens_dict for the selected subset
                - 'router_scores': (B, N) raw router scores for all tokens
                - 'selected_scores': (B, K) normalized scores for selected tokens
        """
        B, N = masked_indices.shape

        # Step A: Compute identity-only embeddings for routing
        identity_emb = self.token_embedding._embed_identity(tokens_dict)  # (B, N, d_identity)

        # During warm-up: use random selection instead of learned routing
        if self.training and self.is_router_warming_up:
            # Random selection matching router's top_k, but always include masked tokens
            top_k = self.router.top_k
            device = masked_indices.device
            
            all_indices = []
            for b in range(B):
                if padding_mask is not None:
                    valid = (~padding_mask[b]).nonzero(as_tuple=True)[0]
                else:
                    valid = torch.arange(N, device=device)
                
                masked_pos = masked_indices[b].nonzero(as_tuple=True)[0]
                n_masked = len(masked_pos)
                
                # Context slots (non-masked valid tokens)
                context_mask = torch.ones(len(valid), dtype=torch.bool, device=device)
                for mp in masked_pos:
                    context_mask[valid == mp] = False
                context_pool = valid[context_mask]
                
                n_context = min(top_k - n_masked, len(context_pool))
                if n_context > 0:
                    perm = torch.randperm(len(context_pool), device=device)[:n_context]
                    context_selected = context_pool[perm]
                else:
                    context_selected = torch.tensor([], dtype=torch.long, device=device)
                
                selected = torch.cat([masked_pos, context_selected])
                # Pad if needed
                if len(selected) < top_k:
                    pad_len = top_k - len(selected)
                    selected = torch.cat([selected, selected[:pad_len].clone()])
                selected = selected[:top_k]
                all_indices.append(selected)
            
            selected_indices = torch.stack(all_indices)  # (B, K)
            normalized_scores = torch.ones(B, top_k, device=device)
            raw_scores = torch.zeros(B, N, device=device)
        else:
            # Learned routing
            selected_indices, normalized_scores, raw_scores = self.router(
                identity_emb, masked_indices, padding_mask
            )

        K = selected_indices.shape[1]

        # Gather the selected subset of tokens_dict
        selected_tokens = self._gather_tokens_dict(tokens_dict, selected_indices)
        selected_masked = torch.gather(masked_indices, 1, selected_indices)  # (B, K)

        # Build padding mask for selected tokens (should be all-valid after routing)
        if padding_mask is not None:
            selected_padding = torch.gather(padding_mask, 1, selected_indices)
        else:
            selected_padding = None

        # Full embedding (identity + value) on the selected subset
        x = self.token_embedding(selected_tokens, selected_masked)  # (B, K, d_model)

        # Step E: Apply relevance score weighting for gradient flow
        # In a generic set transformer, multiplying by normalized scores suppresses context tokens too much.
        # We use a Straight-Through Estimator (STE) trick to pass gradients to the normalized_scores
        # without changing the magnitude of x in the forward pass.
        if not self.is_router_warming_up:
            x = x + (normalized_scores.unsqueeze(-1) - normalized_scores.detach().unsqueeze(-1))

        # Main transformer on the K selected tokens
        out = self.transformer(x, src_key_padding_mask=selected_padding)
        predictions = self.value_head(out)  # (B, K, 1)

        route_info = {
            "selected_indices": selected_indices,
            "selected_masked": selected_masked,
            "selected_tokens": selected_tokens,
            "router_scores": raw_scores,
            "selected_scores": normalized_scores,
        }

        return predictions, route_info
