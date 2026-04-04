import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error

from src.dataloader import load_all_tokens
from src.dataset import hash_str_array, PoolCache
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
    max_targets: int = 512,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading data from {data_dir}...")
    pool = load_all_tokens(data_dir)
    target_date = get_election_date(target_election_id)
    target_type = get_election_type(target_election_id)
    
    print(f"Target Election: {target_election_id} (Type: {target_type}, Date: {target_date})")

    # Build PoolCache
    print("Building PoolCache on GPU...")
    pool_cache = PoolCache(pool, device)

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

    # Sample targets if too many
    if len(target_tokens_idx) > max_targets:
        target_tokens_idx = np.array(random.sample(target_tokens_idx.tolist(), max_targets))

    # Prepare batch tensors
    anchor_dates = torch.tensor([target_date], dtype=torch.float32).to(device)
    target_indices = torch.from_numpy(target_tokens_idx).unsqueeze(0).to(device)  # (1, T)
    target_masked = torch.ones(1, len(target_tokens_idx), dtype=torch.bool).to(device)  # all masked
    target_padding = torch.zeros(1, len(target_tokens_idx), dtype=torch.bool).to(device)

    print("Loading model...")
    model = UniversalMaskedSetTransformer(
        d_model=128, nhead=4, num_layers=4,
        d_router=64, top_k=256, router_warmup_steps=0,
    )
    
    # Handle both old and new checkpoint formats
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()

    # Build key cache
    print("Building key cache...")
    model.rebuild_key_cache(pool_cache)

    print("Evaluating...")
    with torch.no_grad():
        outputs, route_info = model(
            anchor_dates, target_indices, target_masked, target_padding, pool_cache
        )
        
        sel_tokens = route_info["selected_tokens"]
        sel_masked = route_info["selected_masked"]

        # Get predictions for masked tokens
        logits = outputs.squeeze(0).squeeze(-1)  # (K,)
        masked_logits = logits[sel_masked.squeeze(0)]
        pred_scores = torch.softmax(masked_logits, dim=0) * 100.0
        pred_scores_np = pred_scores.cpu().numpy()
        
        # True values
        true_values_np = sel_tokens["values"].squeeze(0)[sel_masked.squeeze(0)].cpu().numpy()
        
        mae = mean_absolute_error(true_values_np, pred_scores_np)
        rmse = np.sqrt(np.mean((true_values_np - pred_scores_np) ** 2))
        
        print("\n--- Results ---")
        n_ctx = (~sel_masked).sum().item()
        n_tgt = sel_masked.sum().item()
        print(f"Evaluated on {n_tgt} target candidate results given {n_ctx} routed context tokens.")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Sum of predicted scores: {pred_scores_np.sum():.2f}%")
        
        print("\nSample Predictions:")
        sort_idx = np.argsort(-true_values_np)
        for i in sort_idx[:min(20, len(true_values_np))]:
            print(f"True: {true_values_np[i]:5.1f}% | Predicted: {pred_scores_np[i]:5.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model on a future election.")
    parser.add_argument("--model", type=str, default="best_model.pth", help="Path to checkpoint")
    parser.add_argument("--target", type=str, default="2024_legi_t1", help="Target election ID")
    parser.add_argument("--max-targets", type=int, default=512, help="Max target tokens to evaluate")
    args = parser.parse_args()
    
    evaluate_future_election(
        model_path=Path(args.model),
        data_dir=Path("data"),
        target_election_id=args.target,
        max_targets=args.max_targets,
    )
