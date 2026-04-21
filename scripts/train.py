import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

SRC_ROOT = PROJECT_ROOT / "src"
sys.path.append(str(SRC_ROOT))

from configs.base_config import (
    TRAIN_IDS_PATH,
    VAL_IDS_PATH,
    VOCAB_PATH,
    BLOCK_SIZE,
    BATCH_SIZE,
    N_EMBD,
    DROPOUT,
    DEVICE,
)
from mini_gpt.tokenizer import CharTokenizer
from mini_gpt.dataset import get_batch
from mini_gpt.model import MiniGPT


def main():
    device = DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. load tokenizer
    tokenizer = CharTokenizer.load(VOCAB_PATH)
    vocab_size = tokenizer.vocab_size
    print(f"Loaded vocab size: {vocab_size}")

    # 2. load processed ids
    train_ids = torch.load(TRAIN_IDS_PATH)
    val_ids = torch.load(VAL_IDS_PATH)
    print(f"Train ids shape: {train_ids.shape}")
    print(f"Val ids shape: {val_ids.shape}")

    # 3. sample one batch
    x, y = get_batch(
        data=train_ids,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        device=device,
    )

    print("\nBatch check:")
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")

    # 4. build model
    model = MiniGPT(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        dropout=DROPOUT,
    ).to(device)

    print("\nModel created successfully.")

    # 5. forward
    logits, loss = model(x, y)

    print("\nForward check:")
    print(f"logits shape: {logits.shape}")
    print(f"loss: {loss.item():.6f}")

    # 6. inspect one position prediction
    probs = torch.softmax(logits[0, 0], dim=-1)
    pred_id = torch.argmax(probs).item()
    pred_char = tokenizer.decode([pred_id])

    print("\nFirst token prediction check:")
    print(f"Predicted token id: {pred_id}")
    print(f"Predicted char: {repr(pred_char)}")


if __name__ == "__main__":
    main()