from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.model import UniversalMaskedSetTransformer


def train_epoch(
    model: UniversalMaskedSetTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    val_dataloader: DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    num_batches = 0
    
    running_train_loss = 0.0
    eval_interval = 10

    # For validation sampling during train
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
            
            # Evaluate a single validation batch
            model.eval()
            try:
                v_tokens_dict, v_masked_batch, v_targets, v_padding_mask = next(val_iter)
            except StopIteration:
                val_iter = iter(val_dataloader)
                v_tokens_dict, v_masked_batch, v_targets, v_padding_mask = next(val_iter)
                
            with torch.no_grad():
                v_padding_mask = v_padding_mask.to(device)
                v_targets = v_targets.to(device)
                v_masked_batch = v_masked_batch.to(device)
                v_tokens_dict = {k: v.to(device) for k, v in v_tokens_dict.items()}

                v_outputs = model(v_tokens_dict, v_masked_batch, v_padding_mask)
                v_flat_outputs = v_outputs.view(-1, 100)
                v_flat_masked = v_masked_batch.view(-1)
                v_masked_outputs = v_flat_outputs[v_flat_masked]
                
                if len(v_targets) > 0:
                    v_loss = criterion(v_masked_outputs, v_targets)
                    writer.add_scalar("Loss/Val/step", v_loss.item(), global_step)
            
            # Switch back to train mode
            model.train()

            print(f"Batch {num_batches}/{len(dataloader)} - Train Loss: {avg_train_loss:.4f} - Val Loss: {(v_loss.item() if len(v_targets) > 0 else 0.0):.4f}", flush=True)

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
) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    num_batches = 0

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

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    writer.add_scalar("Loss/Val/epoch", avg_loss, epoch)
    print(f"Epoch Avg Val Loss: {avg_loss:.4f}", flush=True)
    return avg_loss


def train(
    model: UniversalMaskedSetTransformer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    max_epochs: int,
    learning_rate: float,
    device: torch.device,
    patience: int = 5,
) -> None:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_loss = float("inf")
    patience_counter = 0

    writer = SummaryWriter(log_dir="runs/elections_experiment")

    for epoch in range(max_epochs):
        print(f"Starting epoch {epoch + 1}/{max_epochs}...", flush=True)
        train_loss = train_epoch(model, train_dataloader, optimizer, val_dataloader, device, writer, epoch)
        val_loss = eval_epoch(model, val_dataloader, device, writer, epoch)

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            patience_counter = 0
            print(f"New best val loss: {best_loss:.4f}. Saving checkpoint...", flush=True)
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            print(
                f"No improvement in val loss. Patience: {patience_counter}/{patience}", flush=True
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
    train_dl, val_dl, train_pool, val_pool = build_dataloaders(
        data_dir=data_dir, batch_size=32, max_seq_len=1024, num_workers=16
    )

    print(f"Train pool size: {len(train_pool)} items, Val pool size: {len(val_pool)} items")
    print("Starting training...")
    train(
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        max_epochs=200,
        learning_rate=3e-4,
        device=device,
        patience=10,
    )
    print("Training complete.")
