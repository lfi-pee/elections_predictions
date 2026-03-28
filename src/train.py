from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import UniversalMaskedSetTransformer


def train_epoch(
    model: UniversalMaskedSetTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    num_batches = 0

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
        num_batches += 1
        print(
            f"Batch {num_batches}/{len(dataloader)} Loss: {loss.item():.4f}", flush=True
        )

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"Epoch Avg Loss: {avg_loss:.4f}", flush=True)
    return avg_loss


def train(
    model: UniversalMaskedSetTransformer,
    dataloader: DataLoader,
    max_epochs: int,
    learning_rate: float,
    device: torch.device,
    patience: int = 5,
) -> None:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(max_epochs):
        print(f"Starting epoch {epoch + 1}/{max_epochs}...", flush=True)
        avg_loss = train_epoch(model, dataloader, optimizer, device)

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
            print(f"New best loss: {best_loss:.4f}", flush=True)
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{patience}", flush=True
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}!", flush=True)
            break


if __name__ == "__main__":
    from pathlib import Path
    from src.dataloader import build_training_dataloader

    data_dir = Path("data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    model = UniversalMaskedSetTransformer(d_model=128, nhead=4, num_layers=4)

    print("Building dataloader...")
    # Reduce epoch_size here temporarily for faster debugging, but I'll leave as is
    dataloader = build_training_dataloader(
        data_dir=data_dir, batch_size=32, max_seq_len=1024, epoch_size=100, num_workers=16
    )

    print("Starting training...")
    train(
        model=model,
        dataloader=dataloader,
        max_epochs=200,
        learning_rate=1e-3,
        device=device,
        patience=10,
    )
    print("Training complete.")
