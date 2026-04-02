from __future__ import annotations

from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from src.dataset import TokenDataset, TokenPool, collate_token_sets
from src.load_elections import load_election_tokens
from src.load_polls import load_poll_tokens


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


def load_all_tokens(data_dir: Path) -> TokenPool:
    election_df = load_election_tokens(data_dir)
    poll_df = load_poll_tokens(data_dir)
    poll_df = _resolve_poll_candidates(election_df, poll_df)

    frames = [f for f in [election_df, poll_df] if len(f) > 0]
    combined = pd.concat(frames, ignore_index=True)
    combined.dropna(subset=["value"], inplace=True)
    combined.sort_values("date_float", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return TokenPool(combined)


def build_dataloaders(
    data_dir: Path,
    val_min_date: float = 2025.2916,
    val_election_filter: str = "Municipales",
    dev_fraction: float = 0.05,
    batch_size: int = 32,
    mask_prob: float = 0.15,
    max_seq_len: int = 1024,
    window_years: float = 2.0,
    result_fraction: float = 0.3,
    num_workers: int = 0,
    dev_seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, TokenPool, TokenPool, TokenPool]:
    """Build train / dev / val dataloaders.

    - **val** targets are exclusively 2026 municipales results.
    - **train** targets include ALL other elections (no temporal restriction).
    - Both train and val use the full data pool for context, except that
      2026 municipales Result tokens are removed from train's pool to
      prevent leakage.
    - window_years=2.0 so val muni can see 2024 Europeennes/Legislatives.
    """
    import random as _random

    election_df = load_election_tokens(data_dir)
    poll_df = load_poll_tokens(data_dir)
    poll_df = _resolve_poll_candidates(election_df, poll_df)

    frames = [f for f in [election_df, poll_df] if len(f) > 0]
    combined = pd.concat(frames, ignore_index=True)
    combined.dropna(subset=["value"], inplace=True)
    combined.sort_values("date_float", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # Identify val-only result tokens: 2026 municipales results
    is_val_result = (
        combined["election_type"].str.contains(val_election_filter)
        & (combined["date_float"] > val_min_date)
        & (combined["metric_type"] == "Result")
    )
    print(f"  Val-only result tokens (2026 muni): {is_val_result.sum()} "
          f"out of {len(combined)} total", flush=True)

    # Train pool: everything EXCEPT 2026 muni results (no leakage)
    train_base = combined[~is_val_result].copy().reset_index(drop=True)
    train_pool = TokenPool(train_base)

    # Full pool: everything (val can see all data for context)
    full_pool = TokenPool(combined)

    # --- Split train election groups into train / dev (random) ---
    all_train_groups = list(train_pool.election_groups)
    rng = _random.Random(dev_seed)
    rng.shuffle(all_train_groups)
    n_dev = max(1, int(len(all_train_groups) * dev_fraction))
    dev_groups = all_train_groups[:n_dev]
    train_groups_only = all_train_groups[n_dev:]

    print(f"  Train election groups: {len(all_train_groups)} total  →  "
          f"{len(train_groups_only)} train / {len(dev_groups)} dev", flush=True)

    # --- Val: only 2026 municipales election groups from full pool ---
    val_groups = []
    for grp in full_pool.election_groups:
        anchor_date, result_indices = grp
        if anchor_date > val_min_date:
            et_hash = int(full_pool.election_type[result_indices[0]])
            et_str = full_pool.hash_to_election_type.get(et_hash, "")
            if val_election_filter in et_str:
                val_groups.append(grp)
    print(f"  Val election groups (2026 {val_election_filter}): "
          f"{len(val_groups)}", flush=True)

    # Train dataset: train_pool, visit only train election groups
    train_dataset = TokenDataset(
        pool=train_pool,
        mask_prob=mask_prob,
        max_seq_len=max_seq_len,
        window_years=window_years,
        result_fraction=result_fraction,
        is_training=True,
    )
    train_dataset.election_groups = train_groups_only
    train_dataset.num_elections = len(train_groups_only)
    train_dataset._shuffle_order()

    # Dev dataset: train_pool, visit only dev election groups
    dev_dataset = TokenDataset(
        pool=train_pool,
        mask_prob=mask_prob,
        max_seq_len=max_seq_len,
        window_years=window_years,
        result_fraction=result_fraction,
        is_training=False,
    )
    dev_dataset.election_groups = dev_groups
    dev_dataset.num_elections = len(dev_groups)
    dev_dataset._shuffle_order()

    # Val dataset: full_pool, visit only 2026 muni election groups
    val_dataset = TokenDataset(
        pool=full_pool,
        mask_prob=mask_prob,
        max_seq_len=max_seq_len,
        window_years=window_years,
        result_fraction=result_fraction,
        is_training=False,
    )
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

    return train_dl, dev_dl, val_dl, train_pool, train_pool, full_pool
