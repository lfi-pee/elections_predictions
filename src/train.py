from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import TokenPool
from src.model import UniversalMaskedSetTransformer


def compute_split_metrics(
    tokens_dict: dict[str, torch.Tensor],
    masked_batch: torch.Tensor,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    pool: TokenPool,
) -> dict[str, tuple[float, int]]:
    """Compute per-token loss split by election type and polled/unpolled status.

    Returns a dict mapping split names to (loss_sum, count) tuples.
    """
    criterion = nn.CrossEntropyLoss(reduction="none")
    flat_outputs = outputs.view(-1, 100)
    flat_masked = masked_batch.view(-1)
    masked_outputs = flat_outputs[flat_masked]

    if len(targets) == 0:
        return {}

    per_token_loss = criterion(masked_outputs, targets)  # (N,)

    # Gather the election_type and candidate hashes for masked positions
    flat_election_type = tokens_dict["election_type"].view(-1)[flat_masked]  # (N,)
    flat_candidate = tokens_dict["candidate"].view(-1)[flat_masked]  # (N,)
    flat_metric_type = tokens_dict["metric_type"].view(-1)[flat_masked]  # (N,)

    # Determine "polled" status per masked token:
    # A candidate is "polled" if, within the same batch element, there exists
    # a non-masked token with the same candidate hash and a metric_type that is NOT "Result".
    # We work per batch element.
    B, S = masked_batch.shape
    flat_batch_idx = torch.arange(B, device=masked_batch.device).unsqueeze(1).expand(B, S).reshape(-1)
    masked_batch_idx = flat_batch_idx[flat_masked]  # batch index for each masked token

    hash_to_metric = pool.hash_to_metric_type
    # Build set of (batch_idx, candidate_hash) pairs that have poll data in context
    # i.e., non-masked tokens whose metric_type != Result hash
    result_hashes = {h for h, s in hash_to_metric.items() if s == "Result"}

    non_masked = ~masked_batch  # (B, S)
    flat_non_masked = non_masked.view(-1)
    ctx_batch_idx = flat_batch_idx[flat_non_masked]
    ctx_candidate = tokens_dict["candidate"].view(-1)[flat_non_masked]
    ctx_metric = tokens_dict["metric_type"].view(-1)[flat_non_masked]

    # Filter to poll-type context tokens
    poll_mask_ctx = torch.tensor(
        [int(m.item()) not in result_hashes for m in ctx_metric],
        dtype=torch.bool,
        device=ctx_metric.device,
    )
    poll_ctx_batch = ctx_batch_idx[poll_mask_ctx]
    poll_ctx_cand = ctx_candidate[poll_mask_ctx]

    polled_set: set[tuple[int, int]] = set()
    for b, c in zip(poll_ctx_batch.cpu().tolist(), poll_ctx_cand.cpu().tolist()):
        polled_set.add((b, c))

    # Classify each masked token
    hash_to_election = pool.hash_to_election_type
    splits: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))

    losses_cpu = per_token_loss.detach().cpu().tolist()
    et_cpu = flat_election_type.cpu().tolist()
    cand_cpu = flat_candidate.cpu().tolist()
    bidx_cpu = masked_batch_idx.cpu().tolist()

    for i in range(len(losses_cpu)):
        loss_val = losses_cpu[i]
        etype_hash = et_cpu[i]
        etype_str = hash_to_election.get(etype_hash, f"unk_{etype_hash}")
        is_polled = (bidx_cpu[i], cand_cpu[i]) in polled_set
        poll_str = "polled" if is_polled else "unpolled"

        # By election type
        key_et = etype_str
        s, c = splits[key_et]
        splits[key_et] = (s + loss_val, c + 1)

        # By polled/unpolled
        s, c = splits[poll_str]
        splits[poll_str] = (s + loss_val, c + 1)

        # Granular
        key_gran = f"{etype_str}_{poll_str}"
        s, c = splits[key_gran]
        splits[key_gran] = (s + loss_val, c + 1)

    return dict(splits)


def _eval_one_batch(
    model: UniversalMaskedSetTransformer,
    dl_iter,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    writer: SummaryWriter,
    global_step: int,
    tag: str,
    pool: TokenPool | None,
) -> tuple:
    """Evaluate a single batch from *dataloader* (cycling the iterator).

    Returns (updated_iterator, loss_value_or_None).
    """
    try:
        batch = next(dl_iter)
    except StopIteration:
        dl_iter = iter(dataloader)
        batch = next(dl_iter)

    e_tokens, e_masked, e_targets, e_pad = batch
    e_pad = e_pad.to(device)
    e_targets = e_targets.to(device)
    e_masked = e_masked.to(device)
    e_tokens = {k: v.to(device) for k, v in e_tokens.items()}

    e_outputs = model(e_tokens, e_masked, e_pad)
    e_flat = e_outputs.view(-1, 100)[e_masked.view(-1)]

    if len(e_targets) == 0:
        return dl_iter, None

    e_loss = criterion(e_flat, e_targets)
    writer.add_scalar(f"Loss/{tag}/step", e_loss.item(), global_step)

    if pool is not None:
        for k, (s, c) in compute_split_metrics(e_tokens, e_masked, e_outputs, e_targets, pool).items():
            if c > 0:
                writer.add_scalar(f"Loss/{tag}/{k}", s / c, global_step)

    return dl_iter, e_loss.item()


def train_epoch(
    model: UniversalMaskedSetTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    dev_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    train_pool: TokenPool | None = None,
    dev_pool: TokenPool | None = None,
    val_pool: TokenPool | None = None,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    num_batches = 0
    
    running_train_loss = 0.0
    eval_interval = 10

    dev_iter = iter(dev_dataloader)
    val_iter = iter(val_dataloader)

    for tokens_dict, masked_batch, targets, padding_mask in dataloader:
        optimizer.zero_grad()

        padding_mask = padding_mask.to(device)
        targets = targets.to(device)
        masked_batch = masked_batch.to(device)

        tokens_dict = {k: v.to(device) for k, v in tokens_dict.items()}

        outputs = model(tokens_dict, masked_batch, padding_mask)

        flat_outputs = outputs.view(-1, 100)
        flat_masked = masked_batch.view(-1)

        masked_outputs = flat_outputs[flat_masked]

        if len(targets) == 0:
            continue

        loss = criterion(masked_outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        running_train_loss += loss.item()
        num_batches += 1
        global_step = epoch * len(dataloader) + num_batches
        
        if num_batches % eval_interval == 0:
            avg_train_loss = running_train_loss / eval_interval
            writer.add_scalar("Loss/Train/step", avg_train_loss, global_step)
            running_train_loss = 0.0
            
            # Compute split metrics on this train batch
            if train_pool is not None:
                with torch.no_grad():
                    t_out = model(tokens_dict, masked_batch, padding_mask)
                batch_splits = compute_split_metrics(tokens_dict, masked_batch, t_out, targets, train_pool)
                for k, (s, c) in batch_splits.items():
                    if c > 0:
                        writer.add_scalar(f"Loss/Train/{k}", s / c, global_step)

            # Evaluate one dev batch and one val batch
            model.eval()
            with torch.no_grad():
                dev_iter, d_loss = _eval_one_batch(
                    model, dev_iter, dev_dataloader, device, criterion,
                    writer, global_step, "Dev", dev_pool,
                )
                val_iter, v_loss = _eval_one_batch(
                    model, val_iter, val_dataloader, device, criterion,
                    writer, global_step, "Val", val_pool,
                )
            model.train()

            d_str = f"{d_loss:.4f}" if d_loss is not None else "N/A"
            v_str = f"{v_loss:.4f}" if v_loss is not None else "N/A"
            print(f"Batch {num_batches}/{len(dataloader)} - Train: {avg_train_loss:.4f} - Dev: {d_str} - Val: {v_str}", flush=True)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    writer.add_scalar("Loss/Train/epoch", avg_loss, epoch)
    print(f"Epoch Avg Train Loss: {avg_loss:.4f}", flush=True)
    return avg_loss


@torch.no_grad()
def eval_epoch(
    model: UniversalMaskedSetTransformer,
    dataloader: DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    pool: TokenPool | None = None,
    tag: str = "Dev",
) -> float:
    """Evaluate on a given split.  *tag* controls the TensorBoard prefix."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    num_batches = 0
    split_accum: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))

    for tokens_dict, masked_batch, targets, padding_mask in dataloader:
        padding_mask = padding_mask.to(device)
        targets = targets.to(device)
        masked_batch = masked_batch.to(device)
        tokens_dict = {k: v.to(device) for k, v in tokens_dict.items()}

        outputs = model(tokens_dict, masked_batch, padding_mask)

        flat_outputs = outputs.view(-1, 100)
        flat_masked = masked_batch.view(-1)

        masked_outputs = flat_outputs[flat_masked]

        if len(targets) == 0:
            continue

        loss = criterion(masked_outputs, targets)
        total_loss += loss.item()
        num_batches += 1

        if pool is not None:
            batch_splits = compute_split_metrics(tokens_dict, masked_batch, outputs, targets, pool)
            for k, (s, c) in batch_splits.items():
                prev_s, prev_c = split_accum[k]
                split_accum[k] = (prev_s + s, prev_c + c)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    writer.add_scalar(f"Loss/{tag}/epoch", avg_loss, epoch)

    if pool is not None:
        for k, (s, c) in split_accum.items():
            if c > 0:
                writer.add_scalar(f"Loss/{tag}/{k}", s / c, epoch)
                print(f"  {tag}/{k}: {s / c:.4f} (n={c})", flush=True)

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
    patience: int = 5,
    train_pool: TokenPool | None = None,
    dev_pool: TokenPool | None = None,
    val_pool: TokenPool | None = None,
) -> None:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_loss = float("inf")
    patience_counter = 0

    writer = SummaryWriter(log_dir="runs/elections_experiment")

    for epoch in range(max_epochs):
        print(f"Starting epoch {epoch + 1}/{max_epochs}...", flush=True)
        train_loss = train_epoch(
            model, train_dataloader, optimizer, dev_dataloader, val_dataloader,
            device, writer, epoch,
            train_pool=train_pool, dev_pool=dev_pool, val_pool=val_pool,
        )

        # Dev evaluation — used for early stopping
        dev_loss = eval_epoch(model, dev_dataloader, device, writer, epoch, pool=dev_pool, tag="Dev")

        # Val evaluation — monitoring only (temporal holdout)
        val_loss = eval_epoch(model, val_dataloader, device, writer, epoch, pool=val_pool, tag="Val")

        if dev_loss < best_loss - 1e-4:
            best_loss = dev_loss
            patience_counter = 0
            print(f"New best dev loss: {best_loss:.4f}. Saving checkpoint...", flush=True)
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            print(
                f"No improvement in dev loss. Patience: {patience_counter}/{patience}", flush=True
            )

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

    model = UniversalMaskedSetTransformer(d_model=256, nhead=8, num_layers=8)

    print("Building dataloaders...")
    train_dl, dev_dl, val_dl, train_pool, dev_pool, val_pool = build_dataloaders(
        data_dir=data_dir, batch_size=32, max_seq_len=1024, num_workers=16
    )

    print(f"Train pool size: {len(train_pool)} items, Dev pool size: {len(dev_pool)} items, Val pool size: {len(val_pool)} items")
    print("Starting training...")
    train(
        model=model,
        train_dataloader=train_dl,
        dev_dataloader=dev_dl,
        val_dataloader=val_dl,
        max_epochs=200,
        learning_rate=3e-4,
        device=device,
        patience=10,
        train_pool=train_pool,
        dev_pool=dev_pool,
        val_pool=val_pool,
    )
    print("Training complete.")
