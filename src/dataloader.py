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


def build_dataloaders(
    data_dir: Path,
    split_date: float = 2025.2916,
    batch_size: int = 32,
    mask_prob: float = 0.15,
    max_seq_len: int = 1024,
    window_years: float = 1.0,
    result_fraction: float = 0.3,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, TokenPool, TokenPool]:
    election_df = load_election_tokens(data_dir)
    poll_df = load_poll_tokens(data_dir)

    frames = [f for f in [election_df, poll_df] if len(f) > 0]
    combined = pd.concat(frames, ignore_index=True)
    combined.dropna(subset=["value"], inplace=True)
    combined.sort_values("date_float", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    train_df = combined[combined["date_float"] <= split_date].copy().reset_index(drop=True)
    val_df = combined[combined["date_float"] > split_date].copy().reset_index(drop=True)

    train_pool = TokenPool(train_df)
    val_pool = TokenPool(val_df)

    train_dataset = TokenDataset(
        pool=train_pool,
        mask_prob=mask_prob,
        max_seq_len=max_seq_len,
        window_years=window_years,
        result_fraction=result_fraction,
        is_training=True,
    )

    val_dataset = TokenDataset(
        pool=val_pool,
        mask_prob=mask_prob,
        max_seq_len=max_seq_len,
        window_years=window_years,
        result_fraction=result_fraction,
        is_training=False,
    )

    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_token_sets,
    )

    val_dl = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_token_sets,
    )

    return train_dl, val_dl, train_pool, val_pool
