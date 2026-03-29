from __future__ import annotations

import bisect
import hashlib
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def hash_str_array(arr: np.ndarray, num_buckets: int = 50000) -> np.ndarray:
    def hash_str(s):
        return int(hashlib.md5(str(s).encode("utf-8")).hexdigest(), 16) % num_buckets
    
    codes, uniques = pd.factorize(arr)
    hashed_uniques = np.array([hash_str(str(s)) for s in uniques], dtype=np.int64)
    return hashed_uniques[codes]


class TokenPool:
    def __init__(self, df_sorted: "pd.DataFrame", num_buckets: int = 50000) -> None:
        self.dates = df_sorted["date_float"].values.astype(np.float32)
        
        self.election_type = hash_str_array(df_sorted["election_type"].values, num_buckets)
        self.location = hash_str_array(df_sorted["location"].values, num_buckets)
        self.candidate = hash_str_array(df_sorted["candidate"].values, num_buckets)
        self.party = hash_str_array(df_sorted["party"].values, num_buckets)
        self.metric_type = hash_str_array(df_sorted["metric_type"].values, num_buckets)
        
        self.value = df_sorted["value"].values.astype(np.float32)
        self.is_result = np.array(df_sorted["metric_type"].values == "Result")

        # Geo-coordinates (normalized: centered on France)
        if "latitude" in df_sorted.columns:
            self.latitude = df_sorted["latitude"].values.astype(np.float32)
            self.longitude = df_sorted["longitude"].values.astype(np.float32)
        else:
            self.latitude = np.full(len(self.dates), 46.2276, dtype=np.float32)
            self.longitude = np.full(len(self.dates), 2.2137, dtype=np.float32)

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
        window_half_years: float = 0.5,
        result_fraction: float = 0.3,
        epoch_size: int = 10000,
        is_training: bool = True,
    ) -> None:
        self.pool = pool
        self.mask_prob = mask_prob
        self.max_seq_len = max_seq_len
        self.window_half_years = window_half_years
        self.result_fraction = result_fraction
        self.epoch_size = epoch_size
        self.is_training = is_training
        self.date_min = float(pool.dates[0])
        self.date_max = float(pool.dates[-1])

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, _idx: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        anchor = random.uniform(self.date_min, self.date_max)
        lo, hi = self.pool.indices_in_window(anchor, self.window_half_years)

        window_idx = np.arange(lo, hi)
        window_is_result = self.pool.is_result[lo:hi]
        
        window_results = window_idx[window_is_result]
        window_other = window_idx[~window_is_result]

        n_results = min(int(self.max_seq_len * self.result_fraction), len(window_results))
        n_other = min(self.max_seq_len - n_results, len(window_other))

        if n_results == 0 and len(window_results) > 0:
            n_results = min(1, len(window_results))
            n_other = min(self.max_seq_len - n_results, len(window_other))

        sampled = []
        if n_results > 0:
            sampled.extend(random.sample(window_results.tolist(), n_results))
        if n_other > 0:
            sampled.extend(random.sample(window_other.tolist(), n_other))
        
        if not sampled:
            sampled = [random.randint(0, len(self.pool) - 1)]
        else:
            random.shuffle(sampled)

        sampled_idx = np.array(sampled, dtype=np.int64)

        shift = random.uniform(-10.0, 10.0) if self.is_training else 0.0
        dates = self.pool.dates[sampled_idx] + shift
        
        election_type = self.pool.election_type[sampled_idx]
        location = self.pool.location[sampled_idx]
        candidate = self.pool.candidate[sampled_idx]
        party = self.pool.party[sampled_idx]
        metric_type = self.pool.metric_type[sampled_idx]
        values = self.pool.value[sampled_idx]
        latitude = self.pool.latitude[sampled_idx]
        longitude = self.pool.longitude[sampled_idx]

        seq_len = len(sampled_idx)
        masked = np.random.rand(seq_len) < self.mask_prob
        if not masked.any():
            masked[random.randint(0, seq_len - 1)] = True

        true_values = np.clip(values[masked].astype(np.int64), 0, 99)
        
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
