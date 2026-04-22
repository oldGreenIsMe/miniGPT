import random
import sys
from pathlib import Path
import json

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

SRC_ROOT = PROJECT_ROOT / "src"
sys.path.append(str(SRC_ROOT))

from configs.base_config import (
    TRAIN_IDS_PATH,
    VAL_IDS_PATH,
    VOCAB_PATH,
    BEST_MODEL_PATH,
    BLOCK_SIZE,
    BATCH_SIZE,
    N_EMBD,
    N_HEAD,
    N_LAYER,
    DROPOUT,
    MAX_ITERS,
    EVAL_INTERVAL,
    EVAL_ITERS,
    LEARNING_RATE,
    SEED,
    DEVICE,
    OUTPUT_DIR
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from mini_gpt.tokenizer import CharTokenizer
from mini_gpt.dataset import get_batch
from mini_gpt.model import MiniGPT
from mini_gpt.trainer import estimate_loss

train_loss_history = []
val_loss_history = []
step_history = []


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)

    device = DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = CharTokenizer.load(VOCAB_PATH)
    vocab_size = tokenizer.vocab_size
    print(f"Loaded vocab size: {vocab_size}")

    train_ids = torch.load(TRAIN_IDS_PATH)
    val_ids = torch.load(VAL_IDS_PATH)
    print(f"Train ids shape: {train_ids.shape}")
    print(f"Val ids shape: {val_ids.shape}")

    model = MiniGPT(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("\nStart training...\n")

    for step in range(MAX_ITERS):
        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
            losses = estimate_loss(
                model=model,
                train_ids=train_ids,
                val_ids=val_ids,
                eval_iters=EVAL_ITERS,
                block_size=BLOCK_SIZE,
                batch_size=BATCH_SIZE,
                device=device,
            )

            print(
                f"step {step:4d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f}"
            )

            # ===== 新增：记录 =====
            step_history.append(step)
            train_loss_history.append(losses["train"])
            val_loss_history.append(losses["val"])

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"Saved best model to: {BEST_MODEL_PATH}")

        x, y = get_batch(
            data=train_ids,
            block_size=BLOCK_SIZE,
            batch_size=BATCH_SIZE,
            device=device,
        )

        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    loss_data = {
    "step": step_history,
    "train_loss": train_loss_history,
    "val_loss": val_loss_history,
    }
    loss_path = OUTPUT_DIR / "loss_history.json"

    with open(loss_path, "w") as f:
        json.dump(loss_data, f, indent=2)

    print(f"Saved loss history to: {loss_path}")
    print("\nTraining finished.")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()