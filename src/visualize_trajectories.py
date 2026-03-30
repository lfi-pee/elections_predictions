import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

from src.dataloader import load_all_tokens
from src.dataset import hash_str_array
from src.load_elections import parse_election_id
from src.model import UniversalMaskedSetTransformer

def get_hash(name_or_hash):
    if isinstance(name_or_hash, str):
        return hash_str_array(np.array([name_or_hash.upper()]))[0]
    return name_or_hash

# Set style for premium look
plt.style.use('dark_background')
sns.set_palette("husl")


def predict_trajectory(pool, model, target_election_id, candidate_name_or_hash, device, t_months, exclude_polls=False):
    target_date, target_type = parse_election_id(target_election_id)
    hashed_target_type = hash_str_array(np.array([target_type]))[0]
    candidate_hash = get_hash(candidate_name_or_hash)
    
    # Target tokens (Commune results)
    target_tokens_idx = np.where(
        (np.isclose(pool.dates, target_date, atol=1e-3)) &
        (pool.election_type == hashed_target_type) &
        (pool.candidate == candidate_hash) &
        (pool.is_result == True)
    )[0]
    
    if len(target_tokens_idx) == 0:
        print(f"No target tokens found for {candidate_name_or_hash} in {target_election_id}")
        return None, None, None
        
    np.random.seed(42)
    random.seed(42)
    # We sample up to 200 communes to get a stable national average approximation
    sample_targets = np.random.choice(target_tokens_idx, min(200, len(target_tokens_idx)), replace=False)
    
    preds_mean = []
    preds_std = []
    
    for t in t_months:
        anchor_date = target_date + t / 12.0
        
        # Valid historical context
        context_tokens_idx = np.where(pool.dates <= anchor_date)[0]
        context_tokens_idx = np.setdiff1d(context_tokens_idx, target_tokens_idx) # Remove true target results
        
        if exclude_polls:
            # Exclude polls for this specific candidate completely
            candidate_polls_idx = np.where(
                (pool.candidate == candidate_hash) & 
                (~pool.is_result)
            )[0]
            context_tokens_idx = np.setdiff1d(context_tokens_idx, candidate_polls_idx)
        else:
            # Exclude polls for this candidate that are NOT for the target election round
            candidate_wrong_round_polls_idx = np.where(
                (pool.candidate == candidate_hash) & 
                (~pool.is_result) &
                (pool.election_type != hashed_target_type)
            )[0]
            context_tokens_idx = np.setdiff1d(context_tokens_idx, candidate_wrong_round_polls_idx)
            
        # Limit context to recent years to avoid memory blowup
        recent_context = context_tokens_idx[pool.dates[context_tokens_idx] >= anchor_date - 0.5]
        
        n_targets = len(sample_targets)
        n_context = min(1024 - n_targets, len(recent_context))
        sampled_context = random.sample(recent_context.tolist(), n_context) if len(recent_context) > 0 else []
        sampled_idx = np.array(sample_targets.tolist() + sampled_context, dtype=np.int64)
        
        dates = pool.dates[sampled_idx]
        election_type = pool.election_type[sampled_idx]
        location = pool.location[sampled_idx]
        candidate = pool.candidate[sampled_idx]
        party = pool.party[sampled_idx]
        metric_type = pool.metric_type[sampled_idx]
        values = pool.value[sampled_idx]
        latitude = pool.latitude[sampled_idx]
        longitude = pool.longitude[sampled_idx]
        
        masked = np.isin(sampled_idx, sample_targets)
        
        batched_tokens = {
            "dates": torch.from_numpy(dates).unsqueeze(0).to(device),
            "election_type": torch.from_numpy(election_type).unsqueeze(0).to(device),
            "location": torch.from_numpy(location).unsqueeze(0).to(device),
            "candidate": torch.from_numpy(candidate).unsqueeze(0).to(device),
            "party": torch.from_numpy(party).unsqueeze(0).to(device),
            "metric_type": torch.from_numpy(metric_type).unsqueeze(0).to(device),
            "values": torch.from_numpy(values).unsqueeze(0).to(device),
            "latitude": torch.from_numpy(latitude).unsqueeze(0).to(device),
            "longitude": torch.from_numpy(longitude).unsqueeze(0).to(device),
        }
        masked_batch = torch.from_numpy(masked).unsqueeze(0).to(device)
        padding_mask = torch.zeros((1, len(sampled_idx)), dtype=torch.bool).to(device)
        
        with torch.no_grad():
            outputs = model(batched_tokens, masked_batch, padding_mask)
            flat_outputs = outputs.view(-1, 100)
            masked_outputs = flat_outputs[masked_batch.view(-1)]
            preds_probs = nn.functional.softmax(masked_outputs, dim=1)
            bins = torch.arange(100, dtype=torch.float32, device=device)
            expected_values = (preds_probs * bins).sum(dim=1).cpu().numpy()
            
        preds_mean.append(expected_values.mean())
        preds_std.append(expected_values.std())
        
    true_mean = pool.value[sample_targets].mean()
    
    return np.array(preds_mean), np.array(preds_std), true_mean

def get_actual_polls(pool, target_election_id, candidate_name_or_hash):
    target_date, target_type = parse_election_id(target_election_id)
    hashed_target_type = hash_str_array(np.array([target_type]))[0]
    candidate_hash = get_hash(candidate_name_or_hash)
    
    poll_idx = np.where(
        (pool.candidate == candidate_hash) & 
        (~pool.is_result) & 
        (pool.election_type == hashed_target_type) & 
        (pool.dates >= target_date - 1.0) & # Last 12 months
        (pool.dates <= target_date)
    )[0]
    
    poll_times = (pool.dates[poll_idx] - target_date) * 12.0 # Monthly offset from target date
    poll_values = pool.value[poll_idx]
    
    # Sort by time
    sort_idx = np.argsort(poll_times)
    return poll_times[sort_idx], poll_values[sort_idx]

def get_party_prior_score(pool, target_election_id, candidate_name_or_hash):
    target_date, target_type = parse_election_id(target_election_id)
    hashed_target_type = hash_str_array(np.array([target_type]))[0]
    candidate_hash = get_hash(candidate_name_or_hash)
    
    # Target tokens (Commune results)
    target_tokens_idx = np.where(
        (np.isclose(pool.dates, target_date, atol=1e-3)) &
        (pool.election_type == hashed_target_type) &
        (pool.candidate == candidate_hash) &
        (pool.is_result == True)
    )[0]
    if len(target_tokens_idx) == 0:
        return 1.5

    # The party of this candidate in this election
    party_hash = pool.party[target_tokens_idx[0]]
    
    # Now find party's prior scores
    past_results_idx = np.where(
        (pool.party == party_hash) &
        (pool.is_result == True) &
        (pool.dates < target_date)
    )[0]
    
    if len(past_results_idx) > 0:
        return np.mean(pool.value[past_results_idx])
    return 1.5


def plot_1a_ghost_candidate(pool, model, device, candidate, election_id, split_label, save_dir="."):
    """
    1a. The "Ghost Candidate" Trajectory (Zero-Shot Prediction)
    Converging Line Chart with Confidence Bands
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    t = np.linspace(-12, 0, 30)
    
    expected_value, uncertainty, true_outcome = predict_trajectory(
        pool, model, election_id, candidate, device, t, exclude_polls=True
    )
    
    if expected_value is None:
        print(f"  Skipping ghost candidate plot for {candidate} in {election_id} ({split_label}): no target tokens found")
        plt.close()
        return
    
    prior_score = get_party_prior_score(pool, election_id, candidate)
    baseline = np.full_like(t, prior_score)
    
    _, target_type = parse_election_id(election_id)
    year = election_id.split("_")[0]
    
    ax.plot(t, baseline, linestyle='--', color='gray', linewidth=2, label=f'Naive Baseline (Party Prior: {prior_score:.1f}%)')
    ax.plot(t, np.full_like(t, true_outcome), linestyle=':', color='white', linewidth=2, label='True Final Outcome ({:.1f}%)'.format(true_outcome))
    
    ax.plot(t, expected_value, color='#00d4ff', linewidth=3, label='Universal Model Prediction')
    
    # Fill standard deviation as confidence
    ax.fill_between(t, np.maximum(0, expected_value - uncertainty), expected_value + uncertainty, color='#00d4ff', alpha=0.2, label='Model Variance Across Communes')
    
    ax.set_title(f"1a. Ghost Candidate Trajectory [{split_label.upper()}]\n{target_type} {year} | Candidate: {candidate}", fontsize=15, pad=20, color='white')
    ax.set_xlabel("Months out from Election Day", fontsize=12)
    ax.set_ylabel("Expected Vote Share (%)", fontsize=12)
    ax.set_xlim(-12, 0)
    
    ax.grid(color='#333333', linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=11, facecolor='#111111', edgecolor='#333333')
    
    plt.tight_layout()
    fname = f"viz_1a_ghost_candidate_{split_label}.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}")


def plot_1a_ghost_error_distribution(pool, model, device, election_id, split_label, save_dir="."):
    """
    1a (v2). Average mistake of the model as a function of time (absolute error in pp),
    compared to average party prior error.
    Only takes into account truly ghost candidates with a party affiliation.
    Shows the distribution of error as a function of time.
    """
    print(f"  Generating Ghost Error Distribution [{split_label}]...")
    
    target_date, target_type = parse_election_id(election_id)
    hashed_target_type = hash_str_array(np.array([target_type]))[0]
    year = election_id.split("_")[0]
    
    # All results for this election
    results_idx = np.where(
        (np.isclose(pool.dates, target_date, atol=1e-3)) &
        (pool.election_type == hashed_target_type) &
        (pool.is_result == True)
    )[0]
    
    # Candidate hashes in this election
    unique_candidates = np.unique(pool.candidate[results_idx])
    
    # Find polled candidates
    polls_idx = np.where(
        (pool.election_type == hashed_target_type) &
        (pool.is_result == False) &
        (pool.dates <= target_date)
    )[0]
    polled_candidates = np.unique(pool.candidate[polls_idx])
    
    # "Truly ghost candidates"
    ghost_candidates = np.setdiff1d(unique_candidates, polled_candidates)
    
    # Candidates with party priors
    past_results_idx = np.where(
        (pool.is_result == True) &
        (pool.dates < target_date)
    )[0]
    parties_with_priors = np.unique(pool.party[past_results_idx])
    
    cand_to_party = {c: p for c, p in zip(pool.candidate[results_idx], pool.party[results_idx])}
    
    # Filter candidates without a known affiliated party prior
    valid_ghosts = [c for c in ghost_candidates if cand_to_party.get(c) in parties_with_priors]
    
    if len(valid_ghosts) == 0:
        print(f"  No valid ghost candidates found for {election_id} ({split_label})")
        return
    
    # Subset to keep compute reasonable
    np.random.seed(42)
    selected_ghosts = np.random.choice(valid_ghosts, size=min(15, len(valid_ghosts)), replace=False)
    
    t_months = np.linspace(-12, 0, 10)
    all_model_errors = []
    all_prior_errors = []
    
    for cand in selected_ghosts:
        # Get prior error
        prior_score = get_party_prior_score(pool, election_id, cand)
        
        # Calculate true score for this candidate across the sampled communes
        expected_value, _, true_outcome = predict_trajectory(
            pool, model, election_id, cand, device, t_months, exclude_polls=True
        )
        
        if expected_value is None or np.isnan(true_outcome):
            continue
        
        # Absolute error (in percentage points)
        abs_error_model = np.abs(expected_value - true_outcome)
        abs_error_prior = np.abs(prior_score - true_outcome)
        
        # Store distribution
        all_model_errors.append(abs_error_model)
        all_prior_errors.append(np.full_like(t_months, abs_error_prior))
    
    if len(all_model_errors) == 0:
        print(f"  No valid predictions for ghost error distribution [{split_label}]")
        return
        
    model_errors_matrix = np.array(all_model_errors) # shape: (N_candidates, 10 dates)
    prior_errors_matrix = np.array(all_prior_errors)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 7))
    
    median_model = np.median(model_errors_matrix, axis=0)
    p25_model = np.percentile(model_errors_matrix, 25, axis=0)
    p75_model = np.percentile(model_errors_matrix, 75, axis=0)
    
    median_prior = np.median(prior_errors_matrix, axis=0)
    
    ax.plot(t_months, median_model, color='#00d4ff', linewidth=3, label='Model Median Absolute Error')
    ax.fill_between(t_months, p25_model, p75_model, color='#00d4ff', alpha=0.2, label='Model IQR (25th-75th percentile)')
    
    ax.plot(t_months, median_prior, color='gray', linestyle='--', linewidth=3, label='Party Prior Median Absolute Error')
    
    ax.set_title(f"1a. Model Error vs Prior (Ghost Candidates) [{split_label.upper()}]\n{target_type} {year} | Absolute error (pp)", fontsize=15, pad=20, color='white')
    ax.set_xlabel("Months out from Election Day", fontsize=12)
    ax.set_ylabel("Absolute Error (pp)", fontsize=12)
    ax.set_xlim(-12, 0)
    
    max_y = max(np.max(p75_model), np.max(median_prior))
    ax.set_ylim(0, max_y * 1.25) # Dynamically cap the y-axis so prior is visible
    
    ax.grid(color='#333333', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=11, facecolor='#111111', edgecolor='#333333')
    
    plt.tight_layout()
    fname = f"viz_1a_ghost_error_distribution_{split_label}.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}")


def plot_1b_tracked_candidate(pool, model, device, candidate, election_id, split_label, save_dir="."):
    """
    1b. The "Tracked Candidate" Trajectory (Data-Rich Prediction)
    Signal vs. Noise Smoothing Line Chart
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    t = np.linspace(-12, 0, 30)
    
    expected_value, _, true_outcome = predict_trajectory(
        pool, model, election_id, candidate, device, t, exclude_polls=False
    )
    
    if expected_value is None:
        print(f"  Skipping tracked candidate plot for {candidate} in {election_id} ({split_label}): no target tokens found")
        plt.close()
        return
        
    poll_times, poll_values = get_actual_polls(pool, election_id, candidate)
    
    _, target_type = parse_election_id(election_id)
    year = election_id.split("_")[0]
    
    # Rolling average of raw polls roughly
    if len(poll_times) > 0:
        df = pd.DataFrame({"t": poll_times, "v": poll_values}).sort_values("t")
        df["rolling"] = df["v"].rolling(window=max(1, len(df)//15), min_periods=1).mean()
        ax.scatter(df["t"], df["v"], color='#ff0055', alpha=0.4, s=30, label='Raw Candidate Polls (Noisy)')
        ax.plot(df["t"], df["rolling"], color='gray', linestyle='-', linewidth=2, alpha=0.8, drawstyle='steps-post', label='Naive Rolling Polling Average')
    
    ax.plot(t, np.full_like(t, true_outcome), color='white', linestyle=':', linewidth=2, label='True Final Outcome ({:.1f}%)'.format(true_outcome))
    ax.plot(t, expected_value, color='#00ff88', linewidth=3, label='Universal Model Prediction')
    
    ax.set_title(f"1b. Tracked Candidate Trajectory [{split_label.upper()}]\n{target_type} {year} | Candidate: {candidate}", fontsize=15, pad=20, color='white')
    ax.set_xlabel("Months out from Election Day", fontsize=12)
    ax.set_ylabel("Expected Vote Share (%)", fontsize=12)
    ax.set_xlim(-12, 0)
    
    ax.grid(color='#333333', linestyle='--', alpha=0.5)
    
    ax.legend(loc='lower left', fontsize=11, facecolor='#111111', edgecolor='#333333')
    
    plt.tight_layout()
    fname = f"viz_1b_tracked_candidate_{split_label}.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}")


# ---------------------------------------------------------------
# Split configurations for train/val visualizations
# ---------------------------------------------------------------
SPLIT_CONFIGS = {
    "train": {
        # Ghost candidate: unpolled candidate from a training election
        "ghost_candidate": ("JEAN-JACQUES GAULTIER", "2022_legi_t1"),
        "ghost_error_election": "2022_legi_t1",
        # Tracked candidate: polled candidate from a training election
        "tracked_candidate": ("Marine LE PEN", "2022_pres_t1"),
    },
    "val": {
        # The 2026 Municipales use list-based voting (no individual candidate names),
        # making them incompatible with the candidate-level trajectory visualizations.
        # Use 2024_legi_t1 as the nearest election with candidate-level data.
        "ghost_candidate": None,  # Auto-discover via _find_val_ghost_candidate
        "ghost_error_election": "2024_legi_t1",
        # Tracked: use a well-polled 2024 legislative candidate
        "tracked_candidate": None,  # Will auto-discover
    },
}


def _find_val_ghost_candidate(pool, election_id):
    """Find a valid ghost candidate for the given election — one with a known party
    prior but no direct polling data."""
    target_date, target_type = parse_election_id(election_id)
    hashed_target_type = hash_str_array(np.array([target_type]))[0]
    
    results_idx = np.where(
        (np.isclose(pool.dates, target_date, atol=1e-3)) &
        (pool.election_type == hashed_target_type) &
        (pool.is_result == True)
    )[0]
    
    if len(results_idx) == 0:
        return None
    
    unique_cands = np.unique(pool.candidate[results_idx])
    polls_idx = np.where(
        (pool.election_type == hashed_target_type) &
        (~pool.is_result) &
        (pool.dates <= target_date)
    )[0]
    polled = np.unique(pool.candidate[polls_idx])
    ghost = np.setdiff1d(unique_cands, polled)
    
    past_idx = np.where((pool.is_result == True) & (pool.dates < target_date))[0]
    parties_with_priors = np.unique(pool.party[past_idx])
    cand_to_party = {c: p for c, p in zip(pool.candidate[results_idx], pool.party[results_idx])}
    
    valid = [c for c in ghost if cand_to_party.get(c) in parties_with_priors]
    
    if not valid:
        return None
    
    # Pick a candidate with many commune results (more robust vis)
    counts = {c: np.sum(pool.candidate[results_idx] == c) for c in valid[:50]}
    best = max(counts, key=counts.get)
    return best


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    data_dir = Path("data")
    print("Loading datasets...")
    pool = load_all_tokens(data_dir)
    print(f"Total tokens loaded: {len(pool)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model...")
    model = UniversalMaskedSetTransformer(d_model=256, nhead=8, num_layers=8)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    
    os.makedirs("visualizations", exist_ok=True)
    
    for split_label, cfg in SPLIT_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  Generating visualizations for [{split_label.upper()}] split")
        print(f"{'='*60}")
        
        # 1a. Ghost candidate trajectory
        ghost_cfg = cfg["ghost_candidate"]
        ghost_err_elec = cfg["ghost_error_election"]
        
        if ghost_cfg is not None:
            ghost_cand, ghost_elec = ghost_cfg
        else:
            # Auto-discover a ghost candidate from the ghost error election
            ghost_elec = ghost_err_elec
            ghost_cand_hash = _find_val_ghost_candidate(pool, ghost_elec)
            if ghost_cand_hash is not None:
                ghost_cand = ghost_cand_hash  # Use hash directly
                print(f"  Auto-discovered ghost candidate hash={ghost_cand} for {ghost_elec}")
            else:
                ghost_cand = None
                print(f"  Could not auto-discover ghost candidate for {ghost_elec}")
        
        if ghost_cand is not None:
            print(f"\n  Visualization 1a (Ghost Candidate): {ghost_cand} in {ghost_elec}")
            plot_1a_ghost_candidate(pool, model, device, ghost_cand, ghost_elec, split_label, "visualizations")
        
        # 1a. Ghost error distribution
        print(f"\n  Visualization 1a (Ghost Error Distribution): {ghost_err_elec}")
        plot_1a_ghost_error_distribution(pool, model, device, ghost_err_elec, split_label, "visualizations")
        
        # 1b. Tracked candidate
        tracked = cfg["tracked_candidate"]
        if tracked is not None:
            tracked_cand, tracked_elec = tracked
            print(f"\n  Visualization 1b (Tracked Candidate): {tracked_cand} in {tracked_elec}")
            plot_1b_tracked_candidate(pool, model, device, tracked_cand, tracked_elec, split_label, "visualizations")
        else:
            print(f"\n  Visualization 1b (Tracked Candidate): SKIPPED — no individual candidate-level polls available for {split_label} set elections")
    
    print("\nDone. Saved to 'visualizations/' directory.")

