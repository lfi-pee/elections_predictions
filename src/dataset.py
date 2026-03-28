from __future__ import annotations

import bisect
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data_token import DataToken


class TokenPool:
    def __init__(self, df_sorted: "pd.DataFrame") -> None:
        self.dates = df_sorted["date_float"].values.astype(np.float32)
        self.election_type = df_sorted["election_type"].values
        self.location = df_sorted["location"].values
        self.candidate = df_sorted["candidate"].values
        self.party = df_sorted["party"].values
        self.metric_type = df_sorted["metric_type"].values
        self.value = df_sorted["value"].values.astype(np.float32)
        self.is_result = np.array(self.metric_type == "Result")

    def __len__(self) -> int:
        return len(self.dates)

    def indices_in_window(self, center: float, half_width: float) -> tuple[int, int]:
        lo = bisect.bisect_left(self.dates, center - half_width)
        hi = bisect.bisect_right(self.dates, center + half_width)
        return lo, hi

    def make_token(self, idx: int) -> DataToken:
        return DataToken(
            date_float=float(self.dates[idx]),
            election_type=str(self.election_type[idx]),
            location=str(self.location[idx]),
            candidate=str(self.candidate[idx]),
            party=str(self.party[idx]),
            metric_type=str(self.metric_type[idx]),
            value=float(self.value[idx]),
        )


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

    def __getitem__(
        self, _idx: int
    ) -> tuple[list[DataToken], list[bool], torch.Tensor]:
        anchor = random.uniform(self.date_min, self.date_max)
        lo, hi = self.pool.indices_in_window(anchor, self.window_half_years)

        window_idx = np.arange(lo, hi)
        window_is_result = self.pool.is_result[lo:hi]
        window_results = window_idx[window_is_result].tolist()
        window_other = window_idx[~window_is_result].tolist()

        n_results = min(
            int(self.max_seq_len * self.result_fraction), len(window_results)
        )
        n_other = min(self.max_seq_len - n_results, len(window_other))

        if n_results == 0 and window_results:
            n_results = min(1, len(window_results))
            n_other = min(self.max_seq_len - n_results, len(window_other))

        sampled = []
        if window_results:
            sampled += random.sample(
                window_results, min(n_results, len(window_results))
            )
        if window_other:
            sampled += random.sample(window_other, min(n_other, len(window_other)))
        random.shuffle(sampled)

        if not sampled:
            rand_idx = random.randint(0, len(self.pool) - 1)
            sampled = [rand_idx]

        shift = random.uniform(-10.0, 10.0) if self.is_training else 0.0
        tokens = [self.pool.make_token(i) for i in sampled]
        if shift != 0.0:
            tokens = [
                DataToken(
                    t.date_float + shift,
                    t.election_type,
                    t.location,
                    t.candidate,
                    t.party,
                    t.metric_type,
                    t.value,
                )
                for t in tokens
            ]

        seq_len = len(tokens)
        masked = [random.random() < self.mask_prob for _ in range(seq_len)]
        if not any(masked):
            masked[random.randint(0, seq_len - 1)] = True

        true_values = [
            min(max(int(t.value), 0), 99) for t, m in zip(tokens, masked) if m
        ]
        return tokens, masked, torch.tensor(true_values, dtype=torch.long)


def collate_token_sets(
    batch: list[tuple[list[DataToken], list[bool], torch.Tensor]],
) -> tuple[list[list[DataToken]], list[list[bool]], list[torch.Tensor], torch.Tensor]:
    if not batch:
        return [], [], [], torch.empty((0, 0), dtype=torch.bool)

    max_len = max(len(b[0]) for b in batch)
    pad_token = DataToken(0.0, "", "", "", "", "", 0.0)

    tokens_batch: list[list[DataToken]] = []
    masked_batch: list[list[bool]] = []
    targets_batch: list[torch.Tensor] = []
    padding_mask: list[list[bool]] = []

    for tokens, masks, targets in batch:
        pad_len = max_len - len(tokens)
        tokens_batch.append(tokens + [pad_token] * pad_len)
        masked_batch.append(masks + [False] * pad_len)
        targets_batch.append(targets)
        padding_mask.append([False] * len(tokens) + [True] * pad_len)

    return (
        tokens_batch,
        masked_batch,
        targets_batch,
        torch.tensor(padding_mask, dtype=torch.bool),
    )
