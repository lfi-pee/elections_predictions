from __future__ import annotations

import torch
from torch.nn import CrossEntropyLoss

from src.token import DataToken
from src.model import UniversalMaskedSetTransformer
from src.dataset import TokenDataset, TokenPool, collate_token_sets

import numpy as np
import pandas as pd


def _make_pool(tokens: list[DataToken]) -> TokenPool:
    df = (
        pd.DataFrame(
            [
                {
                    "date_float": np.float32(t.date_float),
                    "election_type": t.election_type,
                    "location": t.location,
                    "candidate": t.candidate,
                    "party": t.party,
                    "metric_type": t.metric_type,
                    "value": np.float32(t.value),
                }
                for t in tokens
            ]
        )
        .sort_values("date_float")
        .reset_index(drop=True)
    )
    return TokenPool(df)


def test_umst_forward_pass_and_loss() -> None:
    tokens = [
        DataToken(
            22.27,
            "Presidentielle_T1",
            "Brest",
            "MACRON Emmanuel",
            "LREM",
            "Result",
            27.8,
        ),
        DataToken(
            22.27, "Presidentielle_T1", "Brest", "Abstention", "", "Context", 26.0
        ),
        DataToken(
            22.27,
            "Presidentielle_T1",
            "Lyon",
            "MACRON Emmanuel",
            "LREM",
            "Result",
            30.1,
        ),
        DataToken(
            22.10,
            "Presidentielle_T1",
            "National",
            "MACRON Emmanuel",
            "LREM",
            "Poll_Ifop",
            28.0,
        ),
    ]
    pool = _make_pool(tokens)
    model = UniversalMaskedSetTransformer(d_model=128, nhead=4, num_layers=2)
    dataset = TokenDataset(
        pool=pool,
        mask_prob=0.5,
        max_seq_len=4,
        window_half_years=1.0,
        epoch_size=2,
        is_training=False,
    )

    samples = [dataset[0], dataset[1]]
    batch_tokens, batch_masks, batch_targets, padding_mask = collate_token_sets(samples)
    predictions = model(batch_tokens, batch_masks, padding_mask)

    flat_masks = [m for seq in batch_masks for m in seq]
    flat_predictions = predictions.view(-1, 100)[flat_masks]
    flat_targets = torch.cat(batch_targets)

    assert flat_predictions.shape == (flat_targets.shape[0], 100)

    loss_fn = CrossEntropyLoss()
    loss = loss_fn(flat_predictions, flat_targets)
    assert loss.item() >= 0.0
    loss.backward()
    assert model.value_head.weight.grad is not None


def test_umst_permutation_invariance() -> None:
    tokens = [
        DataToken(
            22.27,
            "Presidentielle_T1",
            "Brest",
            "MACRON Emmanuel",
            "LREM",
            "Result",
            27.8,
        ),
        DataToken(
            22.27, "Presidentielle_T1", "Brest", "LE PEN Marine", "RN", "Result", 23.0
        ),
        DataToken(
            22.27,
            "Presidentielle_T1",
            "Brest",
            "MELENCHON Jean-Luc",
            "LFI",
            "Result",
            22.0,
        ),
    ]
    model = UniversalMaskedSetTransformer(d_model=128, nhead=4, num_layers=2)
    model.eval()

    with torch.no_grad():
        preds_original = model([tokens], [[True, False, False]])
        tokens_shuffled = [tokens[2], tokens[0], tokens[1]]
        preds_shuffled = model([tokens_shuffled], [[False, True, False]])

    assert torch.allclose(preds_original[0][0], preds_shuffled[0][1], atol=1e-5)
    assert torch.allclose(preds_original[0][1], preds_shuffled[0][2], atol=1e-5)
    assert torch.allclose(preds_original[0][2], preds_shuffled[0][0], atol=1e-5)


def test_variable_length_batching() -> None:
    tokens = [
        DataToken(
            22.27, "Presidentielle_T1", "Brest", "MACRON", "LREM", "Result", 27.8
        ),
        DataToken(
            22.27, "Presidentielle_T1", "Brest", "Abstention", "", "Context", 26.0
        ),
        DataToken(24.43, "Europeennes", "National", "BARDELLA", "RN", "Result", 31.5),
    ]
    pool = _make_pool(tokens)
    model = UniversalMaskedSetTransformer(d_model=128, nhead=4, num_layers=2)
    dataset = TokenDataset(
        pool=pool,
        mask_prob=0.5,
        max_seq_len=2,
        window_half_years=5.0,
        epoch_size=2,
        is_training=False,
    )

    batch_tokens, batch_masks, batch_targets, padding_mask = collate_token_sets(
        [dataset[0], dataset[1]]
    )
    predictions = model(batch_tokens, batch_masks, padding_mask)
    assert predictions.shape[0] == 2
    assert predictions.shape[2] == 100
