from __future__ import annotations

from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from src.dataset import TokenDataset, TokenPool, collate_token_sets
from src.load_elections import load_election_tokens
from src.load_polls import load_poll_tokens


def load_all_tokens(data_dir: Path) -> TokenPool:
    election_df = load_election_tokens(data_dir)
    poll_df = load_poll_tokens(data_dir)

    frames = [f for f in [election_df, poll_df] if len(f) > 0]
    combined = pd.concat(frames, ignore_index=True)
    combined.dropna(subset=["value"], inplace=True)
    combined.sort_values("date_float", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return TokenPool(combined)


def build_training_dataloader(
    data_dir: Path,
    batch_size: int = 32,
    mask_prob: float = 0.15,
    max_seq_len: int = 1024,
    window_half_years: float = 0.5,
    result_fraction: float = 0.3,
    epoch_size: int = 10000,
    num_workers: int = 0,
) -> DataLoader:
    pool = load_all_tokens(data_dir)
    dataset = TokenDataset(
        pool=pool,
        mask_prob=mask_prob,
        max_seq_len=max_seq_len,
        window_half_years=window_half_years,
        result_fraction=result_fraction,
        epoch_size=epoch_size,
        is_training=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_token_sets,
    )
