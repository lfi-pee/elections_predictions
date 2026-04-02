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
    def __init__(self, df_sorted: "pd.DataFrame", num_buckets: int = 50000) -> None:
        self.dates = df_sorted["date_float"].values.astype(np.float32)
        
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
            key = (int(self.election_type[idx]), int(self.location[idx]), float(self.dates[idx]))
            if key not in groups:
                groups[key] = []
            groups[key].append(idx)
        
        # Store as list of (anchor_date, result_indices_array) for fast access
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


class TokenDataset(Dataset):
    def __init__(
        self,
        pool: TokenPool,
        mask_prob: float = 0.15,
        max_seq_len: int = 1024,
        window_years: float = 1.0,
        result_fraction: float = 0.3,
        is_training: bool = True,
        start_date: float | None = None,
        end_date: float | None = None,
    ) -> None:
        self.pool = pool
        self.mask_prob = mask_prob
        self.max_seq_len = max_seq_len
        self.window_years = window_years
        self.result_fraction = result_fraction
        self.is_training = is_training
        self.date_min = float(pool.dates[0])
        self.date_max = float(pool.dates[-1])
        
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

    def __getitem__(self, idx: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        election_idx = self._order[idx % self.num_elections]
        anchor_date, anchor_result_indices = self.election_groups[election_idx]

        # Context window: [anchor_date - window_years, anchor_date]
        lo = bisect.bisect_left(self.pool.dates, anchor_date - self.window_years)
        hi = bisect.bisect_right(self.pool.dates, anchor_date)

        window_idx = np.arange(lo, hi)
        window_is_result = self.pool.is_result[lo:hi]
        
        window_dates = self.pool.dates[lo:hi]
        is_target_elec = window_is_result & (window_dates == anchor_date)
        is_past_result = window_is_result & (window_dates < anchor_date)
        
        target_elec_results = window_idx[is_target_elec]
        past_results = window_idx[is_past_result]
        window_other = window_idx[~window_is_result]
        
        anchor_set = set(anchor_result_indices.tolist())
        other_loc_target_results = [i for i in target_elec_results.tolist() if i not in anchor_set]
        
        masked_target_idxs = anchor_result_indices.tolist()
        unmasked_target_idxs = []
        
        masked_target_idxs = anchor_result_indices.tolist()
        unmasked_target_idxs = []

        if random.random() < 0.75:
            # 1. 75% of samples: one election's location only (no other locations at the same election)
            # 1a. 75% of those: all candidate scores are masked
            # 1b. 25% of those: one or two (if >2 candidates) candidate scores are unmasked
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
            # 2. 25% of samples: results from the same election at another location in the context
            # at most 5 single candidates results at random other places
            if other_loc_target_results:
                n_other_loc = min(5, len(other_loc_target_results))
                other_loc_sampled = random.sample(other_loc_target_results, n_other_loc)
                unmasked_target_idxs.extend(other_loc_sampled)
                
        current_sampled = masked_target_idxs + unmasked_target_idxs
        
        # 3. Past elections context and polling data should be 50/50
        remaining_budget = self.max_seq_len - len(current_sampled)
        if remaining_budget > 0:
            target_past = remaining_budget // 2
            target_other = remaining_budget - target_past
            
            n_past = min(target_past, len(past_results))
            n_other = min(target_other, len(window_other))
            
            # Fill remaining space if one category falls short
            if n_past < target_past:
                n_other = min(len(window_other), n_other + (target_past - n_past))
            elif n_other < target_other:
                n_past = min(len(past_results), n_past + (target_other - n_other))
        else:
            n_past = 0
            n_other = 0
            
        sampled = current_sampled.copy()
        if n_past > 0:
            sampled.extend(random.sample(past_results.tolist(), n_past))
        if n_other > 0:
            sampled.extend(random.sample(window_other.tolist(), n_other))
            
        random.shuffle(sampled)
        sampled_idx = np.array(sampled, dtype=np.int64)

        # Relative date: anchored election is exactly at 0.0
        dates = self.pool.dates[sampled_idx] - anchor_date
        
        election_type = self.pool.election_type[sampled_idx]
        location = self.pool.location[sampled_idx]
        candidate = self.pool.candidate[sampled_idx]
        party = self.pool.party[sampled_idx]
        metric_type = self.pool.metric_type[sampled_idx]
        values = self.pool.value[sampled_idx]
        latitude = self.pool.latitude[sampled_idx]
        longitude = self.pool.longitude[sampled_idx]

        seq_len = len(sampled_idx)
        masked = np.zeros(seq_len, dtype=bool)
        
        masked_set = set(masked_target_idxs)
        unmasked_set = set(unmasked_target_idxs)
        
        for i, token_idx in enumerate(sampled_idx):
            if token_idx in masked_set:
                masked[i] = True
            elif token_idx in unmasked_set:
                masked[i] = False
        is_result_sampled = self.pool.is_result[sampled_idx]
        if not masked.any() and is_result_sampled.any():
            result_locs = np.where(is_result_sampled)[0]
            masked[np.random.choice(result_locs)] = True

        true_values = values[masked].astype(np.float32)
        
        token_dict = {
            "dates": dates,
            "election_type": election_type,
            "location": location,
            "candidate": candidate,
            "party": party,
            "metric_type": metric_type,
            "values": values,
            "latitude": latitude,
            "longitude": longitude,
        }
        return token_dict, masked, true_values


def collate_token_sets(
    batch: list[tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    if not batch:
        return {}, torch.empty((0, 0), dtype=torch.bool), torch.empty(0, dtype=torch.long), torch.empty((0, 0), dtype=torch.bool)

    max_len = max(len(b[1]) for b in batch)
    batch_size = len(batch)
    
    dates = torch.zeros((batch_size, max_len), dtype=torch.float32)
    election_type = torch.zeros((batch_size, max_len), dtype=torch.long)
    location = torch.zeros((batch_size, max_len), dtype=torch.long)
    candidate = torch.zeros((batch_size, max_len), dtype=torch.long)
    party = torch.zeros((batch_size, max_len), dtype=torch.long)
    metric_type = torch.zeros((batch_size, max_len), dtype=torch.long)
    values = torch.zeros((batch_size, max_len), dtype=torch.float32)
    latitude = torch.zeros((batch_size, max_len), dtype=torch.float32)
    longitude = torch.zeros((batch_size, max_len), dtype=torch.float32)
    
    masked_batch = torch.zeros((batch_size, max_len), dtype=torch.bool)
    padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)

    targets = []
    
    for i, (token_dict, masked, target) in enumerate(batch):
        seq_len = len(masked)
        
        dates[i, :seq_len] = torch.from_numpy(token_dict["dates"])
        election_type[i, :seq_len] = torch.from_numpy(token_dict["election_type"])
        location[i, :seq_len] = torch.from_numpy(token_dict["location"])
        candidate[i, :seq_len] = torch.from_numpy(token_dict["candidate"])
        party[i, :seq_len] = torch.from_numpy(token_dict["party"])
        metric_type[i, :seq_len] = torch.from_numpy(token_dict["metric_type"])
        values[i, :seq_len] = torch.from_numpy(token_dict["values"])
        latitude[i, :seq_len] = torch.from_numpy(token_dict["latitude"])
        longitude[i, :seq_len] = torch.from_numpy(token_dict["longitude"])
        
        masked_batch[i, :seq_len] = torch.from_numpy(masked)
        padding_mask[i, :seq_len] = False
        
        targets.append(torch.from_numpy(target))

    batched_tokens = {
        "dates": dates,
        "election_type": election_type,
        "location": location,
        "candidate": candidate,
        "party": party,
        "metric_type": metric_type,
        "values": values,
        "latitude": latitude,
        "longitude": longitude,
    }
    
    return batched_tokens, masked_batch, torch.cat(targets) if targets else torch.empty(0, dtype=torch.long), padding_mask
