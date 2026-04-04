from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import TokenDataset, TokenPool, PoolCache, collate_token_sets
from src.load_elections import load_election_tokens
from src.load_polls import load_poll_tokens
from src.load_demographics import load_demographic_tokens


def _resolve_poll_candidates(election_df: pd.DataFrame, poll_df: pd.DataFrame) -> pd.DataFrame:
    """Resolve poll candidate abbreviations to real candidate names using election results.

    For municipales, election result nuance codes are often coalition-level
    (e.g. LUG, LUD, LUXD) while poll party codes are individual-party level
    (e.g. LFI, LREM, LLR).  We expand each result nuance through
    NUANCE_EQUIVALENCES to build a (location, poll_party) → candidate lookup.
    """
    from src.nuance_mapping import expand_nuance_group

    if len(poll_df) == 0 or len(election_df) == 0:
        return poll_df

    results = election_df[election_df["metric_type"] == "Result"].dropna(
        subset=["location", "party", "candidate"]
    )

    # Build direct lookup (location, party) -> candidate name
    lookup: dict[tuple[str, str], str] = (
        results.groupby(["location", "party"])["candidate"].first().to_dict()
    )

    # Build expanded lookup for municipales: expand coalition nuances
    # so that (location, individual_poll_party) -> candidate
    expanded_lookup: dict[tuple[str, str], str] = {}
    for (loc, nuance), cand in lookup.items():
        for alt_party in expand_nuance_group(nuance):
            key = (loc, alt_party)
            if key not in expanded_lookup:
                expanded_lookup[key] = cand

    def resolve_row(row):
        if "Municipales" in row["election_type"]:
            loc = row["location"]
            party = row["party"]
            # Try direct match first
            key = (loc, party)
            if key in lookup:
                return lookup[key]
            # Try expanded (coalition → individual party) match
            if key in expanded_lookup:
                return expanded_lookup[key]
        return row["candidate"]

    resolved_polls = poll_df.copy()
    resolved_polls["candidate"] = resolved_polls.apply(resolve_row, axis=1)
    return resolved_polls


def _ensure_availability_date(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure availability_date column exists, defaulting to date_float."""
    if "availability_date" not in df.columns:
        df["availability_date"] = df["date_float"]
    return df


def load_all_tokens(data_dir: Path) -> TokenPool:
    election_df = load_election_tokens(data_dir)
    poll_df = load_poll_tokens(data_dir)
    poll_df = _resolve_poll_candidates(election_df, poll_df)
    demo_df = load_demographic_tokens(data_dir)

    frames = [f for f in [election_df, poll_df, demo_df] if len(f) > 0]
    combined = pd.concat(frames, ignore_index=True)
    combined.dropna(subset=["value"], inplace=True)
    combined = _ensure_availability_date(combined)
    combined.sort_values("availability_date", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return TokenPool(combined)


def build_dataloaders(
    data_dir: Path,
    val_min_date: float = 2025.2916,
    val_election_filter: str = "Municipales",
    dev_fraction: float = 0.05,
    batch_size: int = 32,
    num_workers: int = 0,
    eval_num_workers: int = 0,
    dev_seed: int = 42,
    device: torch.device | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, TokenPool, PoolCache]:
    """Build train / dev / val dataloaders and a single PoolCache.

    Uses ONE unified pool for all splits. Val-only tokens (2026 muni results)
    are marked with a mask so the router can suppress them during training.
    """
    import random as _random

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    election_df = load_election_tokens(data_dir)
    poll_df = load_poll_tokens(data_dir)
    poll_df = _resolve_poll_candidates(election_df, poll_df)
    demo_df = load_demographic_tokens(data_dir)

    frames = [f for f in [election_df, poll_df, demo_df] if len(f) > 0]
    combined = pd.concat(frames, ignore_index=True)
    combined.dropna(subset=["value"], inplace=True)
    combined = _ensure_availability_date(combined)
    combined.sort_values("availability_date", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # Identify val-only result tokens: 2026 municipales results
    is_val_result = (
        combined["election_type"].str.contains(val_election_filter)
        & (combined["date_float"] > val_min_date)
        & (combined["metric_type"] == "Result")
    )
    val_only_mask = is_val_result.values
    print(f"  Val-only result tokens (2026 muni): {val_only_mask.sum()} "
          f"out of {len(combined)} total", flush=True)

    # Build unified pool from ALL data
    pool = TokenPool(combined)

    # Build PoolCache on GPU
    print(f"  Building PoolCache on {device}...", flush=True)
    pool_cache = PoolCache(pool, device, val_only_mask=val_only_mask)

    # --- Split election groups into train / dev / val ---
    # Val groups: 2026 municipales result groups
    val_groups = []
    train_all_groups = []
    val_result_indices = set(np.where(val_only_mask)[0].tolist())

    for grp in pool.election_groups:
        anchor_date, result_indices = grp
        # Check if any result index is val-only
        if any(int(i) in val_result_indices for i in result_indices):
            val_groups.append(grp)
        else:
            train_all_groups.append(grp)

    # Split train into train / dev
    rng = _random.Random(dev_seed)
    rng.shuffle(train_all_groups)
    n_dev = max(1, int(len(train_all_groups) * dev_fraction))
    dev_groups = train_all_groups[:n_dev]
    train_groups = train_all_groups[n_dev:]

    print(f"  Train election groups: {len(train_all_groups)} total  →  "
          f"{len(train_groups)} train / {len(dev_groups)} dev", flush=True)
    print(f"  Val election groups (2026 {val_election_filter}): "
          f"{len(val_groups)}", flush=True)

    # Build datasets (simplified — no pool_size, no window)
    train_dataset = TokenDataset(pool=pool, is_training=True)
    train_dataset.election_groups = train_groups
    train_dataset.num_elections = len(train_groups)
    train_dataset._shuffle_order()

    dev_dataset = TokenDataset(pool=pool, is_training=False)
    dev_dataset.election_groups = dev_groups
    dev_dataset.num_elections = len(dev_groups)
    dev_dataset._shuffle_order()

    val_dataset = TokenDataset(pool=pool, is_training=False)
    val_dataset.election_groups = val_groups
    val_dataset.num_elections = len(val_groups)
    val_dataset._shuffle_order()

    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_token_sets,
    )

    dev_dl = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        collate_fn=collate_token_sets,
    )

    val_dl = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        collate_fn=collate_token_sets,
    )

    return train_dl, dev_dl, val_dl, pool, pool_cache
