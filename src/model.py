from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import PoolCache


class StringEmbedding(nn.Module):
    def __init__(self, num_buckets: int, dim: int) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.embedding = nn.Embedding(num_buckets, dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices)


class TokenEmbedding(nn.Module):
    """Token embedding using sinusoidal features + small categorical embeddings.

    Identity = Concat(date_sin, lat_sin, lon_sin, election_emb, candidate_emb, party_emb, metric_emb)
             = 4 + 4 + 4 + 3 + 3 + 3 + 3 = 24 dims
    Value    = Linear(1 → 24) or learned [MASK]
    Total    = 48 dims
    """

    D_EMB = 3        # embedding dim for each categorical
    D_IDENTITY = 24  # 3x4 continuous + 4×3 categorical
    D_VALUE = 24

    def __init__(self, d_model: int = 48, num_buckets: int = 50000) -> None:
        super().__init__()
        assert d_model == self.D_IDENTITY + self.D_VALUE, (
            f"d_model must be {self.D_IDENTITY + self.D_VALUE}, got {d_model}"
        )
        self.d_model = d_model
        self.d_identity = self.D_IDENTITY

        self.value_proj = nn.Linear(1, self.D_VALUE)

        self.election_emb = StringEmbedding(num_buckets, self.D_EMB)
        self.candidate_emb = StringEmbedding(num_buckets, self.D_EMB)
        nn.init.normal_(self.candidate_emb.embedding.weight, mean=0.0, std=0.02)
        self.party_emb = StringEmbedding(num_buckets, self.D_EMB)
        self.metric_emb = StringEmbedding(num_buckets, self.D_EMB)

        self.mask_token = nn.Parameter(torch.randn(self.D_VALUE))
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.3)

    def _embed_identity(self, tokens_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        # Sinusoidal continuous features
        date = tokens_dict["dates"].unsqueeze(-1)                        # (..., 1)
        omega_1 = 2.0 * torch.pi / 40.0
        omega_2 = 2.0 * torch.pi / 8.0
        date_sin = torch.cat([
            torch.sin(omega_1 * date), torch.cos(omega_1 * date),
            torch.sin(omega_2 * date), torch.cos(omega_2 * date)
        ], dim=-1)

        lat = (tokens_dict["latitude"].unsqueeze(-1) - 46.5) / 5.0      # (..., 1)
        lat_sin = torch.cat([
            torch.sin(torch.pi * lat), torch.cos(torch.pi * lat),
            torch.sin(2 * torch.pi * lat), torch.cos(2 * torch.pi * lat)
        ], dim=-1)

        lon = (tokens_dict["longitude"].unsqueeze(-1) - 2.5) / 5.0      # (..., 1)
        lon_sin = torch.cat([
            torch.sin(torch.pi * lon), torch.cos(torch.pi * lon),
            torch.sin(2 * torch.pi * lon), torch.cos(2 * torch.pi * lon)
        ], dim=-1)

        # Small categorical embeddings
        elec = self.election_emb(tokens_dict["election_type"])           # (..., 3)
        cand = self.candidate_emb(tokens_dict["candidate"])              # (..., 3)
        party = self.party_emb(tokens_dict["party"])                     # (..., 3)
        metric = self.metric_emb(tokens_dict["metric_type"])             # (..., 3)

        identity = torch.cat(
            [date_sin, lat_sin, lon_sin, elec, cand, party, metric], dim=-1
        )  # (..., 24)
        return identity

    def _embed_value(
        self, tokens_dict: dict[str, torch.Tensor], masked_indices: torch.Tensor
    ) -> torch.Tensor:
        val_tensor = tokens_dict["values"].unsqueeze(-1)

        val_tensor_masked = torch.where(
            masked_indices.unsqueeze(-1),
            torch.zeros_like(val_tensor),
            val_tensor,
        )
        val_embedded = self.value_proj(val_tensor_masked)

        mask_expanded = masked_indices.unsqueeze(-1)
        val_embedded = torch.where(
            mask_expanded,
            self.mask_token.view(1, 1, -1).expand_as(val_embedded),
            val_embedded,
        )
        return val_embedded

    def forward(
        self, tokens_dict: dict[str, torch.Tensor], masked_indices: torch.Tensor
    ) -> torch.Tensor:
        identity = self._embed_identity(tokens_dict)
        val_embedded = self._embed_value(tokens_dict, masked_indices)

        combined = torch.cat([identity, val_embedded], dim=-1)
        return self.dropout(self.layer_norm(combined))


class LearnableRouter(nn.Module):
    """Full-pool token router using pre-computed key cache.
    
    Scores the ENTIRE token pool against a target-anchor query using
    a pre-computed key cache, then selects the top_k most relevant tokens.
    This eliminates the random-sampling bottleneck and enables the model
    to discover relevant context across the full dataset, including
    geographically distant "twin" locations.
    """

    def __init__(self, d_model: int, d_router: int = 16, top_k: int = 256) -> None:
        super().__init__()
        d_identity = d_model - TokenEmbedding.D_VALUE
        self.d_router = d_router
        self.top_k = top_k

        self.query_proj = nn.Sequential(
            nn.Linear(d_identity, 32),
            nn.GELU(),
            nn.Linear(32, d_router)
        )
        self.key_proj = nn.Sequential(
            nn.Linear(d_identity, 32),
            nn.GELU(),
            nn.Linear(32, d_router)
        )
        self.temperature = nn.Parameter(torch.ones(1))

    @torch.no_grad()
    def build_key_cache(
        self,
        token_embedding: TokenEmbedding,
        pool_cache: PoolCache,
        chunk_size: int = 100_000,
    ) -> None:
        """Pre-compute L2-normalized key projections for the entire pool.
        
        Stores result as float16 on GPU in pool_cache.key_cache.
        """
        N = pool_cache.N
        device = pool_cache.device
        keys = torch.empty(N, self.d_router, dtype=torch.float16, device=device)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            # Build tokens_dict for the chunk (unsqueeze for batch dim)
            chunk_tokens = {
                "dates": pool_cache.dates[start:end].unsqueeze(0),
                "election_type": pool_cache.election_type[start:end].unsqueeze(0),
                "location": pool_cache.location[start:end].unsqueeze(0),
                "candidate": pool_cache.candidate[start:end].unsqueeze(0),
                "party": pool_cache.party[start:end].unsqueeze(0),
                "metric_type": pool_cache.metric_type[start:end].unsqueeze(0),
                "latitude": pool_cache.latitude[start:end].unsqueeze(0),
                "longitude": pool_cache.longitude[start:end].unsqueeze(0),
            }
            identity = token_embedding._embed_identity(chunk_tokens).squeeze(0)  # (chunk, d_identity)
            k = self.key_proj(identity)
            k = F.normalize(k, dim=-1)
            keys[start:end] = k.half()

        pool_cache.key_cache = keys

    def score_and_select(
        self,
        anchor_query: torch.Tensor,
        pool_cache: PoolCache,
        anchor_dates: torch.Tensor,
        target_indices: torch.Tensor,
        target_padding: torch.Tensor,
        is_training: bool = True,
        n_targets_per_sample: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score full pool and select top-K context tokens.
        
        Args:
            anchor_query: (B, d_router) L2-normalized query
            pool_cache: PoolCache with key_cache set
            anchor_dates: (B,) absolute target dates
            target_indices: (B, T) pool indices of target tokens
            target_padding: (B, T) True for padding in target batch
            is_training: if True, suppress val-only tokens
            n_targets_per_sample: (B,) actual number of target tokens per sample
            
        Returns:
            context_indices: (B, K_ctx) selected context pool indices
            context_scores: (B, K_ctx) raw cosine scores for selected context
        """
        B = anchor_query.shape[0]
        device = anchor_query.device

        # Brute-force matmul against full key cache: (B, d) × (d, N) → (B, N)
        raw_scores = torch.mm(
            anchor_query.half(), pool_cache.key_cache.T
        ).float()
        raw_scores = raw_scores / self.temperature.clamp(min=0.1)

        # Mask future tokens: pool is sorted by date, use searchsorted
        cutoffs = torch.searchsorted(pool_cache.dates, anchor_dates)  # (B,)
        for b in range(B):
            raw_scores[b, cutoffs[b]:] = float("-inf")

        # Mask val-only tokens during training
        if is_training and pool_cache.val_only_mask is not None:
            raw_scores[:, pool_cache.val_only_mask] = float("-inf")

        # Mask target tokens (they'll be force-included separately)
        for b in range(B):
            valid_t = target_indices[b][~target_padding[b]]
            raw_scores[b, valid_t] = float("-inf")

        # Select top-K context tokens
        # Budget: top_k minus the number of target tokens per sample
        # Use the minimum budget across the batch for uniform tensor shape
        if n_targets_per_sample is not None:
            max_targets = int(n_targets_per_sample.max().item())
        else:
            max_targets = int((~target_padding).sum(dim=1).max().item())
        
        k_ctx = max(1, self.top_k - max_targets)
        
        _, context_indices = torch.topk(raw_scores, k_ctx, dim=1)  # (B, k_ctx)
        context_scores = torch.gather(raw_scores, 1, context_indices)  # (B, k_ctx)

        return context_indices, context_scores


class UniversalMaskedSetTransformer(nn.Module):
    """Universal Masked Set Transformer with Full-Pool Router.
    
    Scores the entire token pool (16M+) using a pre-computed key cache
    to find the most relevant context tokens, then predicts masked values
    using self-attention on the selected subset.
    """

    def __init__(
        self,
        d_model: int = 48,
        nhead: int = 4,
        num_layers: int = 2,
        d_router: int = 16,
        top_k: int = 256,
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
            dropout=0.3,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.value_head = nn.Linear(d_model, 1)

    @property
    def is_router_warming_up(self) -> bool:
        return self._global_step < self.router_warmup_steps

    @torch.no_grad()
    def rebuild_key_cache(self, pool_cache: PoolCache) -> None:
        """Rebuild the key cache from current model weights."""
        self.router.build_key_cache(self.token_embedding, pool_cache)

    def forward(
        self,
        anchor_dates: torch.Tensor,
        target_indices: torch.Tensor,
        target_masked: torch.Tensor,
        target_padding: torch.Tensor,
        pool_cache: PoolCache,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass with full-pool routing.
        
        Args:
            anchor_dates: (B,) absolute date of each target election
            target_indices: (B, T) pool indices for target tokens
            target_masked: (B, T) True for masked target positions
            target_padding: (B, T) True for padding positions
            pool_cache: PoolCache with pre-computed key_cache
            
        Returns:
            predictions: (B, K, 1) value predictions for all selected tokens
            route_info: dict with routing metadata
        """
        B, T = target_indices.shape
        device = target_indices.device
        n_targets = (~target_padding).sum(dim=1)  # (B,) actual target count

        # --- Step 1: Compute anchor query from target identity embeddings ---
        # Use ABSOLUTE dates for routing (matches key cache)
        target_tokens_abs = pool_cache.gather_tokens(target_indices)
        target_identity = self.token_embedding._embed_identity(target_tokens_abs)  # (B, T, d_identity)
        
        # Pool masked-target embeddings into anchor query
        queries = self.router.query_proj(target_identity)  # (B, T, d_router)
        mask_for_query = (target_masked & ~target_padding).unsqueeze(-1).float()
        mask_count = mask_for_query.sum(dim=1).clamp(min=1.0)
        anchor_query = (queries * mask_for_query).sum(dim=1) / mask_count  # (B, d_router)
        anchor_query = F.normalize(anchor_query, dim=-1)

        # --- Step 2: Select context from the full pool ---
        if self.training and self.is_router_warming_up:
            # Warm-up: random context selection
            k_ctx = max(1, self.router.top_k - int(n_targets.max().item()))
            N = pool_cache.N
            context_indices_list = []
            for b in range(B):
                cutoff = int(torch.searchsorted(pool_cache.dates, anchor_dates[b]).item())
                cutoff = max(cutoff, 1)  # At least 1 token available
                if cutoff < k_ctx:
                    # Not enough past tokens: sample with replacement
                    perm = torch.randint(0, cutoff, (k_ctx,), device=device)
                else:
                    perm = torch.randperm(cutoff, device=device)[:k_ctx]
                context_indices_list.append(perm)
            context_indices = torch.stack(context_indices_list)  # (B, k_ctx)
            context_scores = torch.zeros(B, k_ctx, device=device)
            raw_full_scores = torch.zeros(B, N, device=device)
        else:
            context_indices, context_scores = self.router.score_and_select(
                anchor_query, pool_cache, anchor_dates,
                target_indices, target_padding,
                is_training=self.training,
                n_targets_per_sample=n_targets,
            )
            raw_full_scores = None  # We don't store full (B, N) for memory

        K_ctx = context_indices.shape[1]

        # --- Step 3: Combine targets + context ---
        selected_pool_indices = torch.cat([target_indices, context_indices], dim=1)  # (B, T + K_ctx)
        selected_masked = torch.cat([
            target_masked,
            torch.zeros(B, K_ctx, dtype=torch.bool, device=device),
        ], dim=1)
        selected_padding = torch.cat([
            target_padding,
            torch.zeros(B, K_ctx, dtype=torch.bool, device=device),
        ], dim=1)

        # --- Step 4: Gather tokens and make dates RELATIVE for transformer ---
        selected_tokens = pool_cache.gather_tokens(selected_pool_indices)
        selected_tokens["dates"] = selected_tokens["dates"] - anchor_dates.unsqueeze(1)

        # --- Step 5: Full embedding (identity + value with masking) ---
        x = self.token_embedding(selected_tokens, selected_masked)  # (B, T+K_ctx, d_model)

        # --- Step 6: Live router scores for gradient flow ---
        # Instead of STE perturbation (which causes NaN via log_softmax gradients),
        # compute live scores and pass them through route_info for entropy
        # regularization. The router gets gradient flow through the entropy loss.
        if not self.is_router_warming_up:
            sel_identity = self.token_embedding._embed_identity(selected_tokens)
            sel_keys = self.router.key_proj(sel_identity)  # (B, T+K_ctx, d_router)
            sel_keys = F.normalize(sel_keys, dim=-1)
            anchor_q_expanded = anchor_query.unsqueeze(1)  # (B, 1, d_router)
            live_scores = (anchor_q_expanded * sel_keys).sum(dim=-1)  # (B, T+K_ctx)
            live_scores = live_scores / self.router.temperature.clamp(min=0.1)
        else:
            live_scores = None

        # --- Step 7: Transformer ---
        out = self.transformer(x, src_key_padding_mask=selected_padding)
        predictions = self.value_head(out)  # (B, T+K_ctx, 1)

        route_info = {
            "selected_indices": selected_pool_indices,
            "selected_masked": selected_masked,
            "selected_tokens": selected_tokens,
            "selected_scores": live_scores if live_scores is not None else context_scores,
            "selected_padding": selected_padding,
        }

        return predictions, route_info
