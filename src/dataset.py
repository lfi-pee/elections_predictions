from __future__ import annotations

import bisect
import hashlib
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def hash_str_array(arr: np.ndarray, num_buckets: int = 50000) -> tuple[np.ndarray, dict[int, str]]:
    def hash_str(s):
        return int(hashlib.md5(str(s).encode("utf-8")).hexdigest(), 16) % num_buckets
    
    codes, uniques = pd.factorize(arr)
    hashed_uniques = np.array([hash_str(str(s)) for s in uniques], dtype=np.int64)
    lookup = {int(hashed_uniques[i]): str(uniques[i]) for i in range(len(uniques))}
    return hashed_uniques[codes], lookup


class TokenPool:
    """In-memory token pool sorted by availability_date for causality.

    - ``dates`` (availability_date): when a data point was *published* and
      therefore usable.  The pool is sorted on this column so that
      ``searchsorted`` can efficiently mask future tokens.
    - ``reference_dates`` (date_float): the date the data *describes*.
      This is what the model receives as its temporal input.
    """

    def __init__(self, df_sorted: "pd.DataFrame", num_buckets: int = 50000) -> None:
        # Pool is sorted by availability_date (causality ordering)
        self.dates = df_sorted["availability_date"].values.astype(np.float32)
        self.reference_dates = df_sorted["date_float"].values.astype(np.float32)

        self.election_type, self.hash_to_election_type = hash_str_array(df_sorted["election_type"].values, num_buckets)
        self.location, _ = hash_str_array(df_sorted["location"].values, num_buckets)
        self.candidate, self.hash_to_candidate = hash_str_array(df_sorted["candidate"].values, num_buckets)
        self.party, _ = hash_str_array(df_sorted["party"].values, num_buckets)
        self.metric_type, self.hash_to_metric_type = hash_str_array(df_sorted["metric_type"].values, num_buckets)

        self.value = df_sorted["value"].values.astype(np.float32)
        self.is_result = np.array(df_sorted["metric_type"].values == "Result")

        # Geo-coordinates (normalized: centered on France)
        if "latitude" in df_sorted.columns:
            self.latitude = df_sorted["latitude"].values.astype(np.float32)
            self.longitude = df_sorted["longitude"].values.astype(np.float32)
        else:
            self.latitude = np.full(len(self.dates), 46.2276, dtype=np.float32)
            self.longitude = np.full(len(self.dates), 2.2137, dtype=np.float32)

        # Precompute unique election groups: (election_type, location, date) -> list of result indices
        self._build_election_groups()

    def _build_election_groups(self) -> None:
        result_mask = self.is_result
        result_indices = np.where(result_mask)[0]

        groups: dict[tuple[int, int, float], list[int]] = {}
        for idx in result_indices:
            # Group key uses reference_date (the actual election date)
            key = (int(self.election_type[idx]), int(self.location[idx]), float(self.reference_dates[idx]))
            if key not in groups:
                groups[key] = []
            groups[key].append(idx)

        # anchor_date is the reference_date (actual election date)
        self.election_groups: list[tuple[float, np.ndarray]] = [
            (key[2], np.array(indices, dtype=np.int64))
            for key, indices in groups.items()
        ]

    def __len__(self) -> int:
        return len(self.dates)

    def indices_in_window(self, center: float, half_width: float) -> tuple[int, int]:
        lo = bisect.bisect_left(self.dates, center - half_width)
        hi = bisect.bisect_right(self.dates, center + half_width)
        return lo, hi


class PoolCache:
    """GPU cache of the full token pool for full-pool routing.
    
    Holds all token features as GPU tensors and the pre-computed key cache.
    The key cache is rebuilt periodically by the model to reflect updated weights.
    """

    def __init__(
        self,
        pool: TokenPool,
        device: torch.device,
        val_only_mask: np.ndarray | None = None,
    ) -> None:
        self.device = device
        self.N = len(pool)

        # Causality dates (availability_date) — used for searchsorted masking
        self.dates = torch.from_numpy(pool.dates).to(device)
        # Reference dates (date_float) — used as model temporal input
        self.reference_dates = torch.from_numpy(pool.reference_dates).to(device)

        self.election_type = torch.from_numpy(pool.election_type).to(device)
        self.location = torch.from_numpy(pool.location).to(device)
        self.candidate = torch.from_numpy(pool.candidate).to(device)
        self.party = torch.from_numpy(pool.party).to(device)
        self.metric_type = torch.from_numpy(pool.metric_type).to(device)
        self.values = torch.from_numpy(pool.value).to(device)
        self.latitude = torch.from_numpy(pool.latitude).to(device)
        self.longitude = torch.from_numpy(pool.longitude).to(device)

        # Mask for val-only tokens (True = suppress during training)
        if val_only_mask is not None:
            self.val_only_mask = torch.from_numpy(val_only_mask.astype(bool)).to(device)
        else:
            self.val_only_mask = None

        # Pre-computed key cache — set by model.rebuild_key_cache()
        self.key_cache: torch.Tensor | None = None  # (N, d_router) float16

    def gather_tokens(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        """Gather token features for selected indices.

        The ``"dates"`` entry returns **reference_dates** (what the data
        describes), not availability_dates.  The model should see the
        true temporal coordinate, while causality enforcement is handled
        separately via ``self.dates`` (availability).

        Args:
            indices: (B, K) or (K,) — indices into the pool
        Returns:
            dict of (B, K) or (K,) tensors
        """
        return {
            "dates": self.reference_dates[indices],
            "election_type": self.election_type[indices],
            "location": self.location[indices],
            "candidate": self.candidate[indices],
            "party": self.party[indices],
            "metric_type": self.metric_type[indices],
            "values": self.values[indices],
            "latitude": self.latitude[indices],
            "longitude": self.longitude[indices],
        }


class TokenDataset(Dataset):
    """Simplified dataset that returns only target election group info.
    
    Context selection is handled by the full-pool router in the model,
    not by the dataset. The dataset only decides:
    1. Which election group to predict (the target)
    2. Which tokens in that group are masked vs revealed
    3. Whether to include other-location results from the same election
    """

    def __init__(
        self,
        pool: TokenPool,
        is_training: bool = True,
        start_date: float | None = None,
        end_date: float | None = None,
    ) -> None:
        self.pool = pool
        self.is_training = is_training
        
        valid_groups = []
        for grp in pool.election_groups:
            if start_date is not None and grp[0] <= start_date:
                continue
            if end_date is not None and grp[0] > end_date:
                continue
            valid_groups.append(grp)
            
        self.election_groups = valid_groups
        self.num_elections = len(self.election_groups)
        self._shuffle_order()

    def _shuffle_order(self) -> None:
        """Shuffle the election visit order at the start of each epoch."""
        self._order = list(range(self.num_elections))
        random.shuffle(self._order)

    def __len__(self) -> int:
        return self.num_elections

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        election_idx = self._order[idx % self.num_elections]
        anchor_date, anchor_result_indices = self.election_groups[election_idx]

        masked_target_idxs = anchor_result_indices.tolist()
        unmasked_target_idxs = []

        if random.random() < 0.75:
            # 75%: single location only
            # 25% of those: partial reveal (1-2 candidates unmasked)
            if random.random() < 0.25:
                num_to_reveal = 0
                if len(masked_target_idxs) > 2:
                    num_to_reveal = random.choice([1, 2])
                elif len(masked_target_idxs) == 2:
                    num_to_reveal = 1
                
                if num_to_reveal > 0:
                    unmasked_target_idxs = random.sample(masked_target_idxs, num_to_reveal)
                    masked_target_idxs = [x for x in masked_target_idxs if x not in unmasked_target_idxs]
        else:
            # 25%: include other-location results from the same election date
            lo_same = bisect.bisect_left(self.pool.dates, anchor_date - 1e-6)
            hi_same = bisect.bisect_right(self.pool.dates, anchor_date + 1e-6)
            same_date_mask = self.pool.is_result[lo_same:hi_same]
            same_date_results = np.arange(lo_same, hi_same)[same_date_mask]
            anchor_set = set(anchor_result_indices.tolist())
            other_loc = [i for i in same_date_results.tolist() if i not in anchor_set]
            if other_loc:
                n_other = min(5, len(other_loc))
                unmasked_target_idxs.extend(random.sample(other_loc, n_other))

        return {
            "anchor_date": np.float32(anchor_date),
            "masked_pool_indices": np.array(masked_target_idxs, dtype=np.int64),
            "unmasked_pool_indices": np.array(unmasked_target_idxs, dtype=np.int64),
        }


def collate_token_sets(
    batch: list[dict[str, np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate target election info into batched tensors.
    
    Returns:
        anchor_dates: (B,) absolute dates of target elections
        target_indices: (B, T_max) pool indices for target tokens (masked + unmasked)
        target_masked: (B, T_max) True for masked positions
        target_padding: (B, T_max) True for padding positions
    """
    B = len(batch)
    if B == 0:
        return (
            torch.empty(0, dtype=torch.float32),
            torch.empty((0, 0), dtype=torch.long),
            torch.empty((0, 0), dtype=torch.bool),
            torch.empty((0, 0), dtype=torch.bool),
        )

    anchor_dates = torch.tensor([b["anchor_date"] for b in batch], dtype=torch.float32)

    # Combine masked and unmasked into target arrays
    all_indices = []
    all_is_masked = []
    for b in batch:
        mi = b["masked_pool_indices"]
        ui = b["unmasked_pool_indices"]
        if len(ui) > 0:
            indices = np.concatenate([mi, ui])
            is_masked = np.concatenate([np.ones(len(mi), dtype=bool), np.zeros(len(ui), dtype=bool)])
        else:
            indices = mi
            is_masked = np.ones(len(mi), dtype=bool)
        all_indices.append(indices)
        all_is_masked.append(is_masked)

    max_T = max(len(x) for x in all_indices)

    target_indices = torch.zeros(B, max_T, dtype=torch.long)
    target_masked = torch.zeros(B, max_T, dtype=torch.bool)
    target_padding = torch.ones(B, max_T, dtype=torch.bool)

    for i in range(B):
        n = len(all_indices[i])
        target_indices[i, :n] = torch.from_numpy(all_indices[i])
        target_masked[i, :n] = torch.from_numpy(all_is_masked[i])
        target_padding[i, :n] = False

    return anchor_dates, target_indices, target_masked, target_padding
