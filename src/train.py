from __future__ import annotations

import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import TokenPool, PoolCache
from src.model import UniversalMaskedSetTransformer


class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            self.shadow[k] = v.clone().detach()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Swap model weights with EMA weights. Returns original state."""
        original = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        return original


def compute_router_entropy_loss(
    selected_scores: torch.Tensor,
    selected_masked: torch.Tensor,
    selected_padding: torch.Tensor,
    target_entropy: float = 0.4,
) -> torch.Tensor:
    """Penalize router entropy deviation from a target band.
    
    Computed over the context (non-masked, non-padded) positions' live scores.
    selected_scores, selected_masked, selected_padding all share shape (B, T+K_ctx).
    """
    B = selected_scores.shape[0]
    device = selected_scores.device

    invalid = selected_masked | selected_padding

    entropy_losses = []
    for b in range(B):
        valid = ~invalid[b]
        n_valid = valid.sum()
        if n_valid < 2:
            continue
        scores_b = selected_scores[b][valid].clamp(-30.0, 30.0)
        probs = torch.softmax(scores_b, dim=0)
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        max_entropy = torch.log(n_valid.float())
        norm_entropy = entropy / max_entropy
        entropy_losses.append((norm_entropy - target_entropy) ** 2)

    if not entropy_losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(entropy_losses).mean()


def compute_split_metrics(
    tokens_dict: dict[str, torch.Tensor],
    masked_batch: torch.Tensor,
    outputs: torch.Tensor,
    pool: TokenPool,
) -> dict[str, dict[str, float]]:
    """Compute per-token loss, MAE, and accuracy split by election type."""
    B, S, _ = outputs.shape
    logits = outputs.squeeze(-1)
    values = tokens_dict["values"]
    
    result_hashes = {h for h, s in pool.hash_to_metric_type.items() if s == "Result"}
    metric_type = tokens_dict["metric_type"]
    
    is_result = torch.zeros_like(metric_type, dtype=torch.bool)
    for h in result_hashes:
        is_result |= (metric_type == h)

    per_token_loss = torch.zeros((B, S), device=outputs.device)
    per_token_mae = torch.zeros((B, S), device=outputs.device)
    per_token_acc = torch.zeros((B, S), device=outputs.device)
    
    for b in range(B):
        sample_res_mask = is_result[b]
        if not sample_res_mask.any():
            continue
            
        et = tokens_dict["election_type"][b, sample_res_mask]
        loc = tokens_dict["location"][b, sample_res_mask]
        dt = tokens_dict["dates"][b, sample_res_mask]
        
        unique_keys = torch.stack([et.float(), loc.float(), dt], dim=-1)
        unique_groups, group_indices = torch.unique(unique_keys, dim=0, return_inverse=True)
        
        for g_idx in range(len(unique_groups)):
            mask_g = (group_indices == g_idx)
            g_logits = logits[b, sample_res_mask][mask_g]
            g_targets = values[b, sample_res_mask][mask_g]
            
            t_sum = g_targets.sum()
            if t_sum > 0:
                g_targets = g_targets / t_sum
                g_logits_clamped = g_logits.clamp(-30.0, 30.0)
                log_probs = torch.log_softmax(g_logits_clamped, dim=0)
                g_targets_clamped = g_targets.clamp(min=1e-6)
                g_loss = g_targets_clamped * (torch.log(g_targets_clamped) - log_probs)
                
                pred_probs = torch.softmax(g_logits_clamped, dim=0)
                g_mae = torch.abs(g_targets - pred_probs)
                g_acc = torch.full_like(g_targets, 1.0 if g_targets.argmax() == pred_probs.argmax() else 0.0)
                
                indices_in_sample = torch.where(sample_res_mask)[0][mask_g]
                per_token_loss[b, indices_in_sample] = g_loss
                per_token_mae[b, indices_in_sample] = g_mae
                per_token_acc[b, indices_in_sample] = g_acc

    flat_masked = masked_batch.view(-1)
    masked_losses = per_token_loss.view(-1)[flat_masked]
    masked_mae = per_token_mae.view(-1)[flat_masked]
    masked_acc = per_token_acc.view(-1)[flat_masked]
    
    flat_election_type = tokens_dict["election_type"].view(-1)[flat_masked]
    flat_candidate = tokens_dict["candidate"].view(-1)[flat_masked]

    B, S = masked_batch.shape
    flat_batch_idx = torch.arange(B, device=masked_batch.device).unsqueeze(1).expand(B, S).reshape(-1)
    masked_batch_idx = flat_batch_idx[flat_masked]

    non_masked = ~masked_batch
    flat_non_masked = non_masked.view(-1)
    ctx_batch_idx = flat_batch_idx[flat_non_masked]
    ctx_candidate = tokens_dict["candidate"].view(-1)[flat_non_masked]
    ctx_metric = tokens_dict["metric_type"].view(-1)[flat_non_masked]

    poll_mask_ctx = torch.tensor(
        [int(m.item()) not in result_hashes for m in ctx_metric],
        dtype=torch.bool,
        device=ctx_metric.device,
    )
    poll_ctx_batch = ctx_batch_idx[poll_mask_ctx]
    poll_ctx_cand = ctx_candidate[poll_mask_ctx]

    polled_set = set()
    for b, c in zip(poll_ctx_batch.cpu().tolist(), poll_ctx_cand.cpu().tolist()):
        polled_set.add((b, c))

    hash_to_election = pool.hash_to_election_type
    splits: dict[str, dict[str, float]] = defaultdict(lambda: {"loss": 0.0, "mae": 0.0, "acc": 0.0, "count": 0})

    losses_cpu = masked_losses.detach().cpu().tolist()
    mae_cpu = masked_mae.detach().cpu().tolist()
    acc_cpu = masked_acc.detach().cpu().tolist()
    et_cpu = flat_election_type.cpu().tolist()
    cand_cpu = flat_candidate.cpu().tolist()
    bidx_cpu = masked_batch_idx.cpu().tolist()

    for i in range(len(losses_cpu)):
        loss_val = losses_cpu[i]
        mae_val = mae_cpu[i]
        acc_val = acc_cpu[i]
        etype_hash = et_cpu[i]
        etype_str = hash_to_election.get(etype_hash, f"unk_{etype_hash}")
        is_polled = (bidx_cpu[i], cand_cpu[i]) in polled_set
        poll_str = "polled" if is_polled else "unpolled"

        key_gran = f"{etype_str}_{poll_str}"
        
        for key in [etype_str, poll_str, key_gran]:
            splits[key]["loss"] += loss_val
            splits[key]["mae"] += mae_val
            splits[key]["acc"] += acc_val
            splits[key]["count"] += 1

    return dict(splits)


def compute_election_loss(
    tokens_dict: dict[str, torch.Tensor],
    outputs: torch.Tensor,
    masked_batch: torch.Tensor,
    pool: TokenPool,
) -> tuple[torch.Tensor, float, float]:
    """Group tokens by election and compute softmax cross-entropy loss, MAE, and Accuracy."""
    B, S, _ = outputs.shape
    logits = outputs.squeeze(-1)
    values = tokens_dict["values"]
    
    result_hashes = {h for h, s in pool.hash_to_metric_type.items() if s == "Result"}
    metric_type = tokens_dict["metric_type"]
    is_result = torch.zeros_like(metric_type, dtype=torch.bool)
    for h in result_hashes:
        is_result |= (metric_type == h)
        
    total_loss = 0.0
    total_mae = 0.0
    total_acc = 0.0
    count = 0
    
    for b in range(B):
        sample_res_mask = is_result[b]
        if not sample_res_mask.any():
            continue
            
        sample_logits = logits[b, sample_res_mask]
        sample_values = values[b, sample_res_mask]
        sample_masked = masked_batch[b, sample_res_mask]
        
        et = tokens_dict["election_type"][b, sample_res_mask]
        loc = tokens_dict["location"][b, sample_res_mask]
        dt = tokens_dict["dates"][b, sample_res_mask]
        
        unique_keys = torch.stack([et.float(), loc.float(), dt], dim=-1)
        unique_groups, group_indices = torch.unique(unique_keys, dim=0, return_inverse=True)
        
        for g_idx in range(len(unique_groups)):
            mask_g = (group_indices == g_idx)
            if not sample_masked[mask_g].any():
                continue
                
            g_logits = sample_logits[mask_g]
            g_targets = sample_values[mask_g]
            
            t_sum = g_targets.sum()
            if t_sum > 0:
                g_targets = g_targets / t_sum
                # Clamp logits to prevent overflow in log_softmax
                g_logits_clamped = g_logits.clamp(-30.0, 30.0)
                log_probs = torch.log_softmax(g_logits_clamped, dim=0)
                
                g_masked = sample_masked[mask_g]
                tgt_masked = g_targets[g_masked]
                pred_masked = log_probs[g_masked]
                
                # Clamp targets away from 0 to avoid log(0) explosion
                tgt_clamped = tgt_masked.clamp(min=1e-6)
                loss_g = (tgt_clamped * (torch.log(tgt_clamped) - pred_masked)).sum()
                # Clamp per-group loss to prevent single outlier from destabilizing
                loss_g = loss_g.clamp(max=10.0)
                total_loss += loss_g
                
                pred_probs = torch.softmax(g_logits_clamped, dim=0)
                mae_g = torch.abs(g_targets[g_masked] - pred_probs[g_masked]).mean().item()
                acc_g = 1.0 if g_targets.argmax() == pred_probs.argmax() else 0.0
                
                total_mae += mae_g
                total_acc += acc_g
                count += 1
            
    if count == 0:
        return torch.tensor(0.0, device=outputs.device, requires_grad=True), 0.0, 0.0
    return total_loss / count, total_mae / count, total_acc / count


def _eval_one_batch(
    model: UniversalMaskedSetTransformer,
    dl_iter,
    dataloader: DataLoader,
    device: torch.device,
    pool_cache: PoolCache,
    writer: SummaryWriter,
    global_step: int,
    tag: str,
    pool: TokenPool | None,
) -> tuple:
    """Evaluate a single batch from dataloader (cycling the iterator)."""
    try:
        batch = next(dl_iter)
    except StopIteration:
        dl_iter = iter(dataloader)
        batch = next(dl_iter)

    anchor_dates, target_indices, target_masked, target_padding = batch
    anchor_dates = anchor_dates.to(device)
    target_indices = target_indices.to(device)
    target_masked = target_masked.to(device)
    target_padding = target_padding.to(device)

    if not target_masked.any():
        return dl_iter, None

    outputs, route_info = model(
        anchor_dates, target_indices, target_masked, target_padding, pool_cache
    )

    sel_tokens = route_info["selected_tokens"]
    sel_masked = route_info["selected_masked"]
    e_loss, e_mae, e_acc = compute_election_loss(sel_tokens, outputs, sel_masked, pool)
    writer.add_scalar(f"Loss/{tag}/step", e_loss.item(), global_step)
    writer.add_scalar(f"Metrics/{tag}/MAE_step", e_mae, global_step)
    writer.add_scalar(f"Metrics/{tag}/Accuracy_step", e_acc, global_step)

    if pool is not None:
        for k, metrics in compute_split_metrics(sel_tokens, sel_masked, outputs, pool).items():
            if metrics["count"] > 0:
                writer.add_scalar(f"Loss/{tag}/{k}", metrics["loss"] / metrics["count"], global_step)
                writer.add_scalar(f"Metrics/{tag}/MAE_{k}", metrics["mae"] / metrics["count"], global_step)
                writer.add_scalar(f"Metrics/{tag}/Accuracy_{k}", metrics["acc"] / metrics["count"], global_step)

    return dl_iter, e_loss.item()


def train_epoch(
    model: UniversalMaskedSetTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    dev_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: torch.device,
    pool_cache: PoolCache,
    writer: SummaryWriter,
    epoch: int,
    pool: TokenPool | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ema: EMAModel | None = None,
    entropy_lambda: float = 0.05,
    target_entropy: float = 0.4,
    key_cache_rebuild_interval: int = 100,
    run_name: str = ".",
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    running_train_loss = 0.0
    print_running_train = 0.0
    print_running_dev = 0.0
    print_running_val = 0.0
    print_dev_count = 0
    print_val_count = 0

    eval_interval = 10
    print_interval = 100

    dev_iter = iter(dev_dataloader)
    val_iter = iter(val_dataloader)

    for batch in dataloader:
        anchor_dates, target_indices, target_masked, target_padding = batch
        anchor_dates = anchor_dates.to(device)
        target_indices = target_indices.to(device)
        target_masked = target_masked.to(device)
        target_padding = target_padding.to(device)

        if not target_masked.any():
            continue

        optimizer.zero_grad()

        outputs, route_info = model(
            anchor_dates, target_indices, target_masked, target_padding, pool_cache
        )

        sel_tokens = route_info["selected_tokens"]
        sel_masked = route_info["selected_masked"]
        sel_padding = route_info["selected_padding"]
        loss, mae, acc = compute_election_loss(sel_tokens, outputs, sel_masked, pool)

        # Router entropy regularization
        entropy_loss_val = 0.0
        if not model.is_router_warming_up:
            entropy_loss = compute_router_entropy_loss(
                route_info["selected_scores"], sel_masked, sel_padding,
                target_entropy=target_entropy,
            )
            loss = loss + entropy_lambda * entropy_loss
            entropy_loss_val = entropy_loss.item()

        if not torch.isfinite(loss):
            print(f"  WARNING: Non-finite loss at step {model._global_step}, skipping batch", flush=True)
            optimizer.zero_grad()
            model._global_step += 1
            num_batches += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

        model._global_step += 1

        total_loss += loss.item()
        running_train_loss += loss.item()
        print_running_train += loss.item()
        num_batches += 1
        global_step = epoch * len(dataloader) + num_batches
        
        # Rebuild key cache periodically
        if num_batches % key_cache_rebuild_interval == 0:
            t0 = time.time()
            model.rebuild_key_cache(pool_cache)
            rebuild_time = time.time() - t0
            writer.add_scalar("Router/cache_rebuild_seconds", rebuild_time, global_step)
            print(f"  Key cache rebuilt in {rebuild_time:.1f}s", flush=True)

        if num_batches % eval_interval == 0:
            avg_train_10 = running_train_loss / eval_interval
            writer.add_scalar("Loss/Train/step", avg_train_10, global_step)
            running_train_loss = 0.0

            # Log router statistics
            if not model.is_router_warming_up:
                sel_scores = route_info["selected_scores"]
                
                # Selected tokens stats
                sel_dates = sel_tokens["dates"]
                writer.add_scalar("Router/selected_mean_time_delta", sel_dates.abs().mean().item(), global_step)
                sel_lat = sel_tokens["latitude"]
                sel_lon = sel_tokens["longitude"]
                anchor_mask = (sel_dates == 0.0)
                if anchor_mask.any():
                    anchor_lat = sel_lat[anchor_mask].mean()
                    anchor_lon = sel_lon[anchor_mask].mean()
                    geo_dist = ((sel_lat - anchor_lat)**2 + (sel_lon - anchor_lon)**2).sqrt()
                    writer.add_scalar("Router/selected_mean_geo_dist_km", (geo_dist.mean() * 111.0).item(), global_step)
                writer.add_scalar("Router/temperature", model.router.temperature.item(), global_step)
                writer.add_scalar("Router/entropy_reg_loss", entropy_loss_val, global_step)
            writer.add_scalar("Router/warmup", float(model.is_router_warming_up), global_step)
            if scheduler is not None:
                writer.add_scalar("LR/step", scheduler.get_last_lr()[0], global_step)
            
            # Split metrics on this train batch
            if pool is not None:
                with torch.no_grad():
                    t_out, t_route = model(
                        anchor_dates, target_indices, target_masked, target_padding, pool_cache
                    )
                t_sel_tokens = t_route["selected_tokens"]
                t_sel_masked = t_route["selected_masked"]
                batch_splits = compute_split_metrics(t_sel_tokens, t_sel_masked, t_out, pool)
                for k, metrics in batch_splits.items():
                    if metrics["count"] > 0:
                        writer.add_scalar(f"Loss/Train/{k}", metrics["loss"] / metrics["count"], global_step)
                        writer.add_scalar(f"Metrics/Train/MAE_{k}", metrics["mae"] / metrics["count"], global_step)
                        writer.add_scalar(f"Metrics/Train/Accuracy_{k}", metrics["acc"] / metrics["count"], global_step)

            # Evaluate one dev batch and one val batch
            model.eval()
            with torch.no_grad():
                dev_iter, d_loss = _eval_one_batch(
                    model, dev_iter, dev_dataloader, device, pool_cache,
                    writer, global_step, "Dev", pool,
                )
                val_iter, v_loss = _eval_one_batch(
                    model, val_iter, val_dataloader, device, pool_cache,
                    writer, global_step, "Val", pool,
                )
            model.train()

            if d_loss is not None:
                print_running_dev += d_loss
                print_dev_count += 1
            if v_loss is not None:
                print_running_val += v_loss
                print_val_count += 1

        if num_batches % print_interval == 0:
            avg_train = print_running_train / print_interval
            avg_dev = print_running_dev / print_dev_count if print_dev_count > 0 else 0.0
            avg_val = print_running_val / print_val_count if print_val_count > 0 else 0.0
            
            print(f"Batch {num_batches}/{len(dataloader)} - Train: {avg_train:.4f} - Dev: {avg_dev:.4f} - Val: {avg_val:.4f}", flush=True)
            
            print_running_train = 0.0
            print_running_dev = 0.0
            print_running_val = 0.0
            print_dev_count = 0
            print_val_count = 0

        if num_batches % 1000 == 0:
            ckpt_path = f"{run_name}/checkpoint_epoch{epoch}_step{num_batches}.pth"
            print(f"Saving checkpoint to {ckpt_path}...", flush=True)
            ckpt_data = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "step": num_batches,
                "global_step": model._global_step,
            }
            if scheduler is not None:
                ckpt_data["scheduler"] = scheduler.state_dict()
            if ema is not None:
                ckpt_data["ema"] = ema.shadow
            torch.save(ckpt_data, ckpt_path)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    writer.add_scalar("Loss/Train/epoch", avg_loss, epoch)
    print(f"Epoch Avg Train Loss: {avg_loss:.4f}", flush=True)
    return avg_loss


@torch.no_grad()
def eval_epoch(
    model: UniversalMaskedSetTransformer,
    dataloader: DataLoader,
    device: torch.device,
    pool_cache: PoolCache,
    writer: SummaryWriter,
    epoch: int,
    pool: TokenPool | None = None,
    tag: str = "Dev",
) -> float:
    """Evaluate on a given split."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    split_accum: dict[str, dict[str, float]] = defaultdict(lambda: {"loss": 0.0, "mae": 0.0, "acc": 0.0, "count": 0})

    for batch in dataloader:
        anchor_dates, target_indices, target_masked, target_padding = batch
        anchor_dates = anchor_dates.to(device)
        target_indices = target_indices.to(device)
        target_masked = target_masked.to(device)
        target_padding = target_padding.to(device)

        if not target_masked.any():
            continue

        outputs, route_info = model(
            anchor_dates, target_indices, target_masked, target_padding, pool_cache
        )

        sel_tokens = route_info["selected_tokens"]
        sel_masked = route_info["selected_masked"]
        loss, mae, acc = compute_election_loss(sel_tokens, outputs, sel_masked, pool)
        total_loss += loss.item()
        num_batches += 1

        if pool is not None:
            batch_splits = compute_split_metrics(sel_tokens, sel_masked, outputs, pool)
            for k, metrics in batch_splits.items():
                split_accum[k]["loss"] += metrics["loss"]
                split_accum[k]["mae"] += metrics["mae"]
                split_accum[k]["acc"] += metrics["acc"]
                split_accum[k]["count"] += metrics["count"]

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    writer.add_scalar(f"Loss/{tag}/epoch", avg_loss, epoch)

    if pool is not None:
        for k, metrics in split_accum.items():
            if metrics["count"] > 0:
                c = metrics["count"]
                writer.add_scalar(f"Loss/{tag}/{k}", metrics["loss"] / c, epoch)
                writer.add_scalar(f"Metrics/{tag}/MAE_{k}", metrics["mae"] / c, epoch)
                writer.add_scalar(f"Metrics/{tag}/Accuracy_{k}", metrics["acc"] / c, epoch)
                print(f"  {tag}/{k} | Loss: {metrics['loss'] / c:.4f} | MAE: {metrics['mae'] / c:.4f} | Acc: {metrics['acc'] / c:.4f} (n={c})", flush=True)

    print(f"Epoch Avg {tag} Loss: {avg_loss:.4f}", flush=True)
    return avg_loss


def train(
    model: UniversalMaskedSetTransformer,
    train_dataloader: DataLoader,
    dev_dataloader: DataLoader,
    val_dataloader: DataLoader,
    max_epochs: int,
    learning_rate: float,
    device: torch.device,
    pool: TokenPool,
    pool_cache: PoolCache,
    patience: int = 5,
) -> None:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate,
        fused=torch.cuda.is_available(), weight_decay=0.1,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5000, T_mult=2, eta_min=1e-5,
    )

    ema = EMAModel(model, decay=0.999)

    # Build initial key cache
    print("Building initial key cache...", flush=True)
    t0 = time.time()
    model.rebuild_key_cache(pool_cache)
    print(f"Key cache built in {time.time() - t0:.1f}s "
          f"({pool_cache.key_cache.shape}, {pool_cache.key_cache.dtype})", flush=True)

    best_loss = float("inf")
    patience_counter = 0

    import os
    from datetime import datetime
    run_name = f"runs/elections_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(run_name, exist_ok=True)
    writer = SummaryWriter(log_dir=run_name)

    for epoch in range(max_epochs):
        print(f"Starting epoch {epoch + 1}/{max_epochs}...", flush=True)
        train_loss = train_epoch(
            model, train_dataloader, optimizer, dev_dataloader, val_dataloader,
            device, pool_cache, writer, epoch,
            pool=pool, scheduler=scheduler, ema=ema, run_name=run_name,
        )

        # Evaluate using EMA weights
        original_state = ema.apply(model)
        model.rebuild_key_cache(pool_cache)  # Rebuild with EMA weights for eval
        dev_loss = eval_epoch(model, dev_dataloader, device, pool_cache, writer, epoch, pool=pool, tag="Dev")
        val_loss = eval_epoch(model, val_dataloader, device, pool_cache, writer, epoch, pool=pool, tag="Val")
        model.load_state_dict(original_state)
        model.rebuild_key_cache(pool_cache)  # Rebuild with live weights

        if dev_loss < best_loss - 1e-4:
            best_loss = dev_loss
            patience_counter = 0
            best_path = f"{run_name}/best_model.pth"
            print(f"New best dev loss: {best_loss:.4f}. Saving EMA checkpoint to {best_path}...", flush=True)
            torch.save(ema.shadow, best_path)
        else:
            patience_counter += 1
            print(f"No improvement in dev loss. Patience: {patience_counter}/{patience}", flush=True)

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}!", flush=True)
            break
            
    writer.close()


if __name__ == "__main__":
    from pathlib import Path
    from src.dataloader import build_dataloaders

    data_dir = Path("data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    model = UniversalMaskedSetTransformer(
        d_model=48, nhead=4, num_layers=2,
        d_router=16, top_k=256, router_warmup_steps=500,
    )

    print("Building dataloaders...")
    train_dl, dev_dl, val_dl, pool, pool_cache = build_dataloaders(
        data_dir=data_dir, batch_size=32,
        num_workers=4, eval_num_workers=2,
        device=device,
    )

    print(f"Pool size: {len(pool)} items")
    print("Starting training...")
    train(
        model=model,
        train_dataloader=train_dl,
        dev_dataloader=dev_dl,
        val_dataloader=val_dl,
        max_epochs=200,
        learning_rate=1e-3,
        device=device,
        pool=pool,
        pool_cache=pool_cache,
        patience=10,
    )
    print("Training complete.")
