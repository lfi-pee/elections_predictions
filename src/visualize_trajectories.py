import os
import random
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

from src.dataloader import load_all_tokens
from src.dataset import hash_str_array, PoolCache
from src.load_elections import parse_election_id
from src.model import UniversalMaskedSetTransformer

def get_hash(name_or_hash):
    if isinstance(name_or_hash, str):
        return hash_str_array(np.array([name_or_hash.upper()]))[0]
    return name_or_hash

def get_commune_weights(data_dir: Path):
    print("Loading commune weights from general_results.parquet...")
    df = pd.read_parquet(
        data_dir / "elections" / "agregees" / "general_results.parquet",
        columns=["id_election", "code_commune", "inscrits"]
    )
    def hash_loc(s):
        return int(hashlib.md5(str(s).encode("utf-8")).hexdigest(), 16) % 50000
    
    uniques = df["code_commune"].unique()
    hash_map = {x: hash_loc(x) for x in uniques}
    df["loc_hash"] = df["code_commune"].map(hash_map)
    return df.groupby(["id_election", "loc_hash"])["inscrits"].sum().to_dict()

plt.style.use('dark_background')
sns.set_palette("husl")

def predict_trajectory(pool, model, target_election_id, candidate_name_or_hash, device, t_months, weights_dict, pool_cache=None, exclude_polls=False):
    target_date, target_type = parse_election_id(target_election_id)
    hashed_target_type = hash_str_array(np.array([target_type]))[0]
    candidate_hash = get_hash(candidate_name_or_hash)
    
    target_tokens_idx = np.where(
        (np.isclose(pool.dates, target_date, atol=1e-3)) &
        (pool.election_type == hashed_target_type) &
        (pool.candidate == candidate_hash) &
        (pool.is_result == True)
    )[0]
    
    if len(target_tokens_idx) == 0:
        return {"france_mean": None}
        
    all_election_results_idx = np.where(
        (np.isclose(pool.dates, target_date, atol=1e-1)) & 
        (pool.election_type == hashed_target_type) &
        (pool.is_result == True)
    )[0]
    
    locs = pool.location[all_election_results_idx]
    unique_locs = np.unique(locs)
    
    np.random.seed(42)
    random.seed(42)
    sample_locs = np.random.choice(unique_locs, min(50, len(unique_locs)), replace=False)
    
    sample_targets = all_election_results_idx[np.isin(locs, sample_locs)]
    
    preds_france_mean = []
    preds_std = []
    preds_matrix = []
    
    for t in t_months:
        anchor_date = target_date + t / 12.0
        
        # Build target batch for the model's new API
        anchor_dates_t = torch.tensor([anchor_date], dtype=torch.float32).to(device)
        target_indices_t = torch.from_numpy(sample_targets).unsqueeze(0).to(device)  # (1, T)
        target_masked_t = torch.ones(1, len(sample_targets), dtype=torch.bool).to(device)
        target_padding_t = torch.zeros(1, len(sample_targets), dtype=torch.bool).to(device)
        
        with torch.no_grad():
            outputs, route_info = model(
                anchor_dates_t, target_indices_t, target_masked_t, target_padding_t, pool_cache
            )
            logits = outputs.squeeze(0).squeeze(-1)
            
            sel_masked = route_info["selected_masked"].squeeze(0)
            sel_locs = route_info["selected_tokens"]["location"].squeeze(0)
            
            res_logits = logits[sel_masked]
            res_locs = sel_locs[sel_masked]
            unique_res_locs = torch.unique(res_locs)
            
            candidate_scores = {}
            for l in unique_res_locs:
                loc_mask = (res_locs == l)
                loc_logits = res_logits[loc_mask]
                loc_probs = torch.softmax(loc_logits, dim=0) * 100.0
                
                commune_targets_idx = sample_targets[np.isin(pool.location[sample_targets], l.cpu().item())]
                commune_is_my_cand = (pool.candidate[commune_targets_idx] == candidate_hash)
                
                if commune_is_my_cand.any():
                    score = loc_probs[np.where(commune_is_my_cand)[0][0]]
                    candidate_scores[l.item()] = score.item()
            
            step_preds = []
            step_weights = []
            for loc in sample_locs:
                if loc in candidate_scores:
                    step_preds.append(candidate_scores[loc])
                    w = weights_dict.get((target_election_id, loc), 1.0)
                    step_weights.append(w)
                else:
                    step_preds.append(np.nan)
                    step_weights.append(0.0)
                    
            preds_matrix.append(step_preds)
            
            valid = ~np.isnan(step_preds)
            if valid.any():
                v_preds = np.array(step_preds)[valid]
                v_weights = np.array(step_weights)[valid]
                wm = np.average(v_preds, weights=v_weights) if v_weights.sum() > 0 else v_preds.mean()
                preds_france_mean.append(wm)
                preds_std.append(v_preds.std())
            else:
                preds_france_mean.append(0.0)
                preds_std.append(0.0)
                
    true_scores = []
    true_weights = []
    for loc in sample_locs:
        idx = sample_targets[(pool.location[sample_targets] == loc) & (pool.candidate[sample_targets] == candidate_hash)]
        if len(idx) > 0:
            true_scores.append(pool.value[idx[0]])
            w = weights_dict.get((target_election_id, loc), 1.0)
            true_weights.append(w)
        else:
            true_scores.append(np.nan)
            true_weights.append(0.0)
            
    v = ~np.isnan(true_scores)
    if v.any():
        v_true = np.array(true_scores)[v]
        v_w = np.array(true_weights)[v]
        true_france = np.average(v_true, weights=v_w) if v_w.sum() > 0 else v_true.mean()
    else:
        true_france = 0.0
        
    return {
        "france_mean": np.array(preds_france_mean),
        "uncertainty": np.array(preds_std),
        "france_true": true_france,
        "commune_preds": np.array(preds_matrix),
        "commune_trues": np.array(true_scores)
    }

def get_actual_polls(pool, target_election_id, candidate_name_or_hash):
    target_date, target_type = parse_election_id(target_election_id)
    hashed_target_type = hash_str_array(np.array([target_type]))[0]
    candidate_hash = get_hash(candidate_name_or_hash)
    poll_idx = np.where(
        (pool.candidate == candidate_hash) & 
        (~pool.is_result) & 
        (pool.election_type == hashed_target_type) & 
        (pool.dates >= target_date - 1.0) &
        (pool.dates <= target_date)
    )[0]
    poll_times = (pool.dates[poll_idx] - target_date) * 12.0
    poll_values = pool.value[poll_idx]
    sort_idx = np.argsort(poll_times)
    return poll_times[sort_idx], poll_values[sort_idx]

def get_party_prior_score(pool, target_election_id, candidate_name_or_hash, weights_dict):
    target_date, target_type = parse_election_id(target_election_id)
    hashed_target_type = hash_str_array(np.array([target_type]))[0]
    candidate_hash = get_hash(candidate_name_or_hash)
    
    target_tokens_idx = np.where(
        (np.isclose(pool.dates, target_date, atol=1e-3)) &
        (pool.election_type == hashed_target_type) &
        (pool.candidate == candidate_hash) &
        (pool.is_result == True)
    )[0]
    if len(target_tokens_idx) == 0:
        return 1.5

    party_hash = pool.party[target_tokens_idx[0]]
    past_results_idx = np.where(
        (pool.party == party_hash) &
        (pool.is_result == True) &
        (pool.dates < target_date)
    )[0]
    
    if len(past_results_idx) > 0:
        w_list = [weights_dict.get((target_election_id, loc), 1.0) for loc in pool.location[past_results_idx]]
        w = np.array(w_list)
        v = pool.value[past_results_idx]
        return np.average(v, weights=w) if w.sum() > 0 else np.mean(v)
    return 1.5

def plot_1a_ghost_candidate(pool, model, device, candidate, election_id, split_label, weights_dict, save_dir=".", pool_cache=None):
    t = np.linspace(-12, 0, 30)
    res = predict_trajectory(pool, model, election_id, candidate, device, t, weights_dict, pool_cache=pool_cache, exclude_polls=True)
    
    if res["france_mean"] is None or len(res["france_mean"]) == 0:
        return
    
    prior_score = get_party_prior_score(pool, election_id, candidate, weights_dict)
    baseline = np.full_like(t, prior_score)
    _, target_type = parse_election_id(election_id)
    year = election_id.split("_")[0]
    
    # France (National) Level Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(t, baseline, linestyle='--', color='gray', linewidth=2, label=f'Naive Baseline (Party Prior: {prior_score:.1f}%)')
    ax.plot(t, np.full_like(t, res["france_true"]), linestyle=':', color='white', linewidth=2, label='True Final Outcome ({:.1f}%)'.format(res["france_true"]))
    ax.plot(t, res["france_mean"], color='#00d4ff', linewidth=3, label='Model Prediction (France Weighted)')
    ax.fill_between(t, np.maximum(0, res["france_mean"] - res["uncertainty"]), res["france_mean"] + res["uncertainty"], color='#00d4ff', alpha=0.2, label='Model Variance Across Communes')
    
    ax.set_title(f"1a. Ghost Candidate Trajectory [FRANCE LEVEL] [{split_label.upper()}]\n{target_type} {year} | Candidate: {candidate}", fontsize=15, pad=20, color='white')
    ax.set_xlabel("Months out from Election Day", fontsize=12)
    ax.set_ylabel("Expected Vote Share (%)", fontsize=12)
    ax.set_xlim(-12, 0)
    ax.grid(color='#333333', linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=11, facecolor='#111111', edgecolor='#333333')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"viz_1a_ghost_candidate_national_{split_label}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Commune Level Plot
    valid_c_idx = np.where(~np.isnan(res["commune_trues"]))[0]
    if len(valid_c_idx) > 0:
        c_idx = valid_c_idx[0]
        c_true = res["commune_trues"][c_idx]
        c_preds = res["commune_preds"][:, c_idx]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(t, baseline, linestyle='--', color='gray', linewidth=2, label=f'Naive Baseline (Party Prior: {prior_score:.1f}%)')
        ax.plot(t, np.full_like(t, c_true), linestyle=':', color='white', linewidth=2, label='True Local Outcome ({:.1f}%)'.format(c_true))
        ax.plot(t, c_preds, color='#ff00aa', linewidth=3, label='Model Prediction (Single Commune)')
        
        ax.set_title(f"1a. Ghost Candidate Trajectory [COMMUNE LEVEL] [{split_label.upper()}]\n{target_type} {year} | Candidate: {candidate}", fontsize=15, pad=20, color='white')
        ax.set_xlabel("Months out from Election Day", fontsize=12)
        ax.set_ylabel("Expected Local Vote Share (%)", fontsize=12)
        ax.set_xlim(-12, 0)
        ax.grid(color='#333333', linestyle='--', alpha=0.5)
        ax.legend(loc='upper left', fontsize=11, facecolor='#111111', edgecolor='#333333')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"viz_1a_ghost_candidate_commune_{split_label}.png"), dpi=300, bbox_inches='tight')
        plt.close()

def plot_1a_ghost_error_distribution(pool, model, device, election_id, split_label, weights_dict, save_dir=".", pool_cache=None):
    target_date, target_type = parse_election_id(election_id)
    hashed_target_type = hash_str_array(np.array([target_type]))[0]
    year = election_id.split("_")[0]
    
    results_idx = np.where(
        (np.isclose(pool.dates, target_date, atol=1e-3)) &
        (pool.election_type == hashed_target_type) &
        (pool.is_result == True)
    )[0]
    unique_candidates = np.unique(pool.candidate[results_idx])
    
    polls_idx = np.where(
        (pool.election_type == hashed_target_type) &
        (pool.is_result == False) &
        (pool.dates <= target_date)
    )[0]
    polled_candidates = np.unique(pool.candidate[polls_idx])
    ghost_candidates = np.setdiff1d(unique_candidates, polled_candidates)
    
    past_results_idx = np.where((pool.is_result == True) & (pool.dates < target_date))[0]
    parties_with_priors = np.unique(pool.party[past_results_idx])
    cand_to_party = {c: p for c, p in zip(pool.candidate[results_idx], pool.party[results_idx])}
    
    valid_ghosts = [c for c in ghost_candidates if cand_to_party.get(c) in parties_with_priors]
    if len(valid_ghosts) == 0:
        return
        
    np.random.seed(42)
    selected_ghosts = np.random.choice(valid_ghosts, size=min(15, len(valid_ghosts)), replace=False)
    
    t_months = np.linspace(-12, 0, 10)
    all_model_errors_france = []
    all_model_errors_commune = []
    all_prior_errors = []
    
    for cand in selected_ghosts:
        prior_score = get_party_prior_score(pool, election_id, cand, weights_dict)
        res = predict_trajectory(pool, model, election_id, cand, device, t_months, weights_dict, pool_cache=pool_cache, exclude_polls=True)
        
        if res["france_mean"] is None or np.isnan(res["france_true"]):
            continue
            
        abs_error_model_france = np.abs(res["france_mean"] - res["france_true"])
        abs_error_prior = np.abs(prior_score - res["france_true"])
        
        valid_c_mask = ~np.isnan(res["commune_trues"])
        if valid_c_mask.any():
            c_preds = res["commune_preds"][:, valid_c_mask]
            c_trues = res["commune_trues"][valid_c_mask]
            abs_error_model_commune = np.nanmean(np.abs(c_preds - c_trues), axis=1)
        else:
            abs_error_model_commune = np.full_like(abs_error_model_france, np.nan)
            
        all_model_errors_france.append(abs_error_model_france)
        all_model_errors_commune.append(abs_error_model_commune)
        all_prior_errors.append(np.full_like(t_months, abs_error_prior))
    
    if len(all_model_errors_france) == 0:
        return
        
    median_model_france = np.nanmedian(np.array(all_model_errors_france), axis=0)
    median_model_commune = np.nanmedian(np.array(all_model_errors_commune), axis=0)
    median_prior = np.nanmedian(np.array(all_prior_errors), axis=0)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(t_months, median_model_france, color='#00d4ff', linewidth=3, label='France Level Model MEA')
    ax.plot(t_months, median_model_commune, color='#ff00aa', linewidth=3, linestyle='-.', label='Commune Level Model MEA')
    ax.plot(t_months, median_prior, color='gray', linestyle='--', linewidth=3, label='Party Prior MEA')
    
    ax.set_title(f"1a. Model Error Granularity Comparison (Ghost Candidates) [{split_label.upper()}]\n{target_type} {year} | Absolute error (pp)", fontsize=15, pad=20, color='white')
    ax.set_xlabel("Months out from Election Day", fontsize=12)
    ax.set_ylabel("Median Absolute Error (pp)", fontsize=12)
    ax.set_xlim(-12, 0)
    ax.set_ylim(0, max(np.max(median_model_commune), np.max(median_prior)) * 1.25)
    
    ax.grid(color='#333333', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=11, facecolor='#111111', edgecolor='#333333')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"viz_1a_ghost_error_distribution_{split_label}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_1b_tracked_candidate(pool, model, device, candidate, election_id, split_label, weights_dict, save_dir=".", pool_cache=None):
    t = np.linspace(-12, 0, 30)
    res = predict_trajectory(pool, model, election_id, candidate, device, t, weights_dict, pool_cache=pool_cache, exclude_polls=False)
    
    if res["france_mean"] is None or len(res["france_mean"]) == 0:
        return
        
    poll_times, poll_values = get_actual_polls(pool, election_id, candidate)
    _, target_type = parse_election_id(election_id)
    year = election_id.split("_")[0]
    
    # France (National) Level Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    if len(poll_times) > 0:
        df = pd.DataFrame({"t": poll_times, "v": poll_values}).sort_values("t")
        df["rolling"] = df["v"].rolling(window=max(1, len(df)//15), min_periods=1).mean()
        ax.scatter(df["t"], df["v"], color='#ff0055', alpha=0.4, s=30, label='Raw Candidate Polls (Noisy)')
        ax.plot(df["t"], df["rolling"], color='gray', linestyle='-', linewidth=2, alpha=0.8, drawstyle='steps-post', label='Naive Rolling Polling Average')
    
    ax.plot(t, np.full_like(t, res["france_true"]), color='white', linestyle=':', linewidth=2, label='True Final Outcome ({:.1f}%)'.format(res["france_true"]))
    ax.plot(t, res["france_mean"], color='#00ff88', linewidth=3, label='Universal Model Prediction (France)')
    
    ax.set_title(f"1b. Tracked Candidate Trajectory [FRANCE LEVEL] [{split_label.upper()}]\n{target_type} {year} | Candidate: {candidate}", fontsize=15, pad=20, color='white')
    ax.set_xlabel("Months out from Election Day", fontsize=12)
    ax.set_ylabel("Expected Vote Share (%)", fontsize=12)
    ax.set_xlim(-12, 0)
    ax.grid(color='#333333', linestyle='--', alpha=0.5)
    ax.legend(loc='lower left', fontsize=11, facecolor='#111111', edgecolor='#333333')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"viz_1b_tracked_candidate_national_{split_label}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Commune Level Plot
    valid_c_idx = np.where(~np.isnan(res["commune_trues"]))[0]
    if len(valid_c_idx) > 0:
        c_idx = valid_c_idx[0]
        c_true = res["commune_trues"][c_idx]
        c_preds = res["commune_preds"][:, c_idx]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(t, np.full_like(t, c_true), color='white', linestyle=':', linewidth=2, label='True Local Outcome ({:.1f}%)'.format(c_true))
        ax.plot(t, c_preds, color='#ff00aa', linewidth=3, label='Universal Model Prediction (Single Commune)')
        
        ax.set_title(f"1b. Tracked Candidate Trajectory [COMMUNE LEVEL] [{split_label.upper()}]\n{target_type} {year} | Candidate: {candidate}", fontsize=15, pad=20, color='white')
        ax.set_xlabel("Months out from Election Day", fontsize=12)
        ax.set_ylabel("Expected Local Vote Share (%)", fontsize=12)
        ax.set_xlim(-12, 0)
        ax.grid(color='#333333', linestyle='--', alpha=0.5)
        ax.legend(loc='lower left', fontsize=11, facecolor='#111111', edgecolor='#333333')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"viz_1b_tracked_candidate_commune_{split_label}.png"), dpi=300, bbox_inches='tight')
        plt.close()

SPLIT_CONFIGS = {
    "train": {
        "ghost_candidate": ("JEAN-JACQUES GAULTIER", "2022_legi_t1"),
        "ghost_error_election": "2022_legi_t1",
        "tracked_candidate": ("Marine LE PEN", "2022_pres_t1"),
    },
    "val": {
        "ghost_candidate": None,
        "ghost_error_election": "2024_legi_t1",
        "tracked_candidate": None,
    },
}

def _find_val_ghost_candidate(pool, election_id):
    target_date, target_type = parse_election_id(election_id)
    hashed_target_type = hash_str_array(np.array([target_type]))[0]
    results_idx = np.where((np.isclose(pool.dates, target_date, atol=1e-3)) & (pool.election_type == hashed_target_type) & (pool.is_result == True))[0]
    if len(results_idx) == 0: return None
    unique_cands = np.unique(pool.candidate[results_idx])
    polls_idx = np.where((pool.election_type == hashed_target_type) & (~pool.is_result) & (pool.dates <= target_date))[0]
    polled = np.unique(pool.candidate[polls_idx])
    ghost = np.setdiff1d(unique_cands, polled)
    past_idx = np.where((pool.is_result == True) & (pool.dates < target_date))[0]
    parties_with_priors = np.unique(pool.party[past_idx])
    cand_to_party = {c: p for c, p in zip(pool.candidate[results_idx], pool.party[results_idx])}
    valid = [c for c in ghost if cand_to_party.get(c) in parties_with_priors]
    if not valid: return None
    counts = {c: np.sum(pool.candidate[results_idx] == c) for c in valid[:50]}
    return max(counts, key=counts.get)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    import glob
    
    data_dir = Path("data")
    pool = load_all_tokens(data_dir)
    weights_dict = get_commune_weights(data_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniversalMaskedSetTransformer(d_model=128, nhead=4, num_layers=4, d_router=64, top_k=256, router_warmup_steps=0)
    
    ckpt_files = glob.glob("checkpoint_epoch*_step*.pth")
    if not ckpt_files: ckpt_files = glob.glob("best_model.pth")
    if ckpt_files:
        latest_ckpt = max(ckpt_files, key=os.path.getmtime)
        ckpt = torch.load(latest_ckpt, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
    
    model.to(device)
    model.eval()
    
    # Build PoolCache and key cache
    print("Building PoolCache...")
    pc = PoolCache(pool, device)
    model.rebuild_key_cache(pc)
    
    os.makedirs("visualizations", exist_ok=True)
    
    for split_label, cfg in SPLIT_CONFIGS.items():
        ghost_cfg = cfg["ghost_candidate"]
        ghost_err_elec = cfg["ghost_error_election"]
        if ghost_cfg is not None:
            ghost_cand, ghost_elec = ghost_cfg
        else:
            ghost_elec = ghost_err_elec
            ghost_cand_hash = _find_val_ghost_candidate(pool, ghost_elec)
            ghost_cand = ghost_cand_hash if ghost_cand_hash is not None else None
        
        if ghost_cand is not None:
            plot_1a_ghost_candidate(pool, model, device, ghost_cand, ghost_elec, split_label, weights_dict, "visualizations", pool_cache=pc)
        
        plot_1a_ghost_error_distribution(pool, model, device, ghost_err_elec, split_label, weights_dict, "visualizations", pool_cache=pc)
        
        tracked = cfg["tracked_candidate"]
        if tracked is not None:
            tracked_cand, tracked_elec = tracked
            plot_1b_tracked_candidate(pool, model, device, tracked_cand, tracked_elec, split_label, weights_dict, "visualizations", pool_cache=pc)
