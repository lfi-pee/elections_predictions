import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error

from src.dataloader import load_all_tokens
from src.dataset import hash_str_array
from src.load_elections import ELECTION_TYPE_LABEL, ELECTION_MONTH, parse_election_id
from src.model import UniversalMaskedSetTransformer


def get_election_date(id_election: str) -> float:
    return parse_election_id(id_election)[0]


def get_election_type(id_election: str) -> str:
    return parse_election_id(id_election)[1]


def evaluate_future_election(
    model_path: Path,
    data_dir: Path,
    target_election_id: str,
    context_half_years: float = 0.5,
    max_seq_len: int = 2048,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading data from {data_dir}...")
    pool = load_all_tokens(data_dir)
    target_date = get_election_date(target_election_id)
    target_type = get_election_type(target_election_id)
    
    print(f"Target Election: {target_election_id} (Type: {target_type}, Date: {target_date})")

    # Hash the target type to match pool format
    hashed_target_type = hash_str_array(np.array([target_type]))[0]

    # Find the indices of the target election tokens
    target_tokens_idx = np.where(
        (np.isclose(pool.dates, target_date, atol=1e-3)) &
        (pool.election_type == hashed_target_type) &
        (pool.is_result == True)
    )[0]

    if len(target_tokens_idx) == 0:
        print("No target tokens found for this election!")
        return

    print(f"Found {len(target_tokens_idx)} tokens for the target election.")

    # Find context tokens: strict past
    # Context window: [target_date - context_half_years * 2, target_date]
    # But ONLY tokens strictly before the target date, OR tokens of the target election that we'll mask
    
    # We want a context window extending x years into the past
    start_date = target_date - (context_half_years * 2)
    
    context_tokens_idx = np.where(
        (pool.dates >= start_date) & 
        (pool.dates <= target_date)
    )[0]
    
    # Remove the target token indices from the context exactly, to manipulate them separately
    context_tokens_idx = np.setdiff1d(context_tokens_idx, target_tokens_idx)
    
    # In case there are too many context tokens, sample them
    context_tokens_list = context_tokens_idx.tolist()
    target_tokens_list = target_tokens_idx.tolist()
    
    n_targets = min(max_seq_len // 4, len(target_tokens_list))
    n_context = min(max_seq_len - n_targets, len(context_tokens_list))
    
    sampled_targets = random.sample(target_tokens_list, n_targets)
    sampled_context = random.sample(context_tokens_list, n_context)
    
    sampled_idx = np.array(sampled_targets + sampled_context, dtype=np.int64)
    np.random.shuffle(sampled_idx)

    # Reconstruct token dictionary
    dates = pool.dates[sampled_idx]
    election_type = pool.election_type[sampled_idx]
    location = pool.location[sampled_idx]
    candidate = pool.candidate[sampled_idx]
    party = pool.party[sampled_idx]
    metric_type = pool.metric_type[sampled_idx]
    values = pool.value[sampled_idx]
    latitude = pool.latitude[sampled_idx]
    longitude = pool.longitude[sampled_idx]

    # Identify which tokens are our targets (the ones we need to mask)
    masked = np.isin(sampled_idx, sampled_targets)
    true_values = values[masked].astype(np.float32)

    seq_len = len(sampled_idx)
    
    # Convert to batched tensors
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
    padding_mask = torch.zeros((1, seq_len), dtype=torch.bool).to(device)
    targets = torch.from_numpy(true_values).to(device)

    print("Loading model...")
    model = UniversalMaskedSetTransformer(d_model=256, nhead=8, num_layers=8)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("Evaluating...")
    with torch.no_grad():
        outputs = model(batched_tokens, masked_batch, padding_mask)
        # outputs: (1, S, 1) -> (S,)
        logits = outputs.squeeze(0).squeeze(-1)
        
        # We apply softmax ONLY across the candidates of the target election
        # to ensure they sum to 100%.
        target_mask_tensor = torch.from_numpy(masked).to(device)
        target_logits = logits[target_mask_tensor]
        
        # Predicted percentages (0-100)
        pred_scores = torch.softmax(target_logits, dim=0) * 100.0
        pred_scores_np = pred_scores.cpu().numpy()
        
        true_values_np = targets.cpu().numpy()
        
        mae = mean_absolute_error(true_values_np, pred_scores_np)
        rmse = np.sqrt(np.mean((true_values_np - pred_scores_np) ** 2))
        
        print("\n--- Results ---")
        print(f"Evaluated on {n_targets} target candidate results given {n_context} context tokens.")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Sum of predicted scores: {pred_scores_np.sum():.2f}%")
        
        print("\nSample Predictions:")
        # Sort by true value for better readability
        sort_idx = np.argsort(-true_values_np)
        for i in sort_idx[:min(20, len(true_values_np))]:
            print(f"True: {true_values_np[i]:5.1f}% | Predicted: {pred_scores_np[i]:5.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model on a future election.")
    parser.add_argument("--model", type=str, default="best_model.pth", help="Path to checkpoint")
    parser.add_argument("--target", type=str, default="2024_legi_t1", help="Target election ID")
    parser.add_argument("--context", type=float, default=0.5, help="Context size in years")
    parser.add_argument("--seq-len", type=int, default=1024, help="Max sequence length")
    args = parser.parse_args()
    
    evaluate_future_election(
        model_path=Path(args.model),
        data_dir=Path("data"),
        target_election_id=args.target,
        context_half_years=args.context,
        max_seq_len=args.seq_len
    )
