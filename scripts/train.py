import sys
from pathlib import Path

import torch

# 让 Python 能找到 src/ 里的包
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 把项目根目录加入 sys.path，这样可以找到 configs
sys.path.append(str(PROJECT_ROOT))

# 把 src 目录加入 sys.path，这样可以找到 mini_gpt 包
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.append(str(SRC_ROOT))

from configs.base_config import (
    RAW_TEXT_PATH,
    VOCAB_PATH,
    TRAIN_IDS_PATH,
    VAL_IDS_PATH,
    TRAIN_RATION,
    BLOCK_SIZE,
    BATCH_SIZE,
    DEVICE
)
from mini_gpt.tokenizer import CharTokenizer
from mini_gpt.dataset import build_train_val_ids, get_batch


def validate_sequence_length(data, block_size, split_name):
    if len(data) < block_size + 1:
        raise ValueError(
            f"{split_name} is too short for block_size={block_size}. "
            f"Got len({split_name})={len(data)}, but need at least {block_size + 1}."
        )
    

def main():
    device = DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. read raw text
    with open(RAW_TEXT_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    
    if len(text) == 0:
        raise ValueError("Input text is empty. Please check data/raw/input.txt")

    print(f"Total characters in text: {len(text)}")

    # 2. build tokenizer
    tokenizer = CharTokenizer.from_text(text)
    print(f"Vocab size: {tokenizer.vocab_size}")

    tokenizer.save(VOCAB_PATH)
    print(f"Saved vocab to: {VOCAB_PATH}")

    # 3. build train/val ids
    train_ids, val_ids = build_train_val_ids(
        tokenizer=tokenizer,
        text=text,
        train_ratio=TRAIN_RATION,
    )

    TRAIN_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train_ids, TRAIN_IDS_PATH)
    torch.save(val_ids, VAL_IDS_PATH)

    print(f"Train ids shape: {train_ids.shape}")
    print(f"Val ids shape: {val_ids.shape}")
    print(f"Saved train ids to: {TRAIN_IDS_PATH}")
    print(f"Saved val ids to: {VAL_IDS_PATH}")

    # 4. validate lengths
    validate_sequence_length(train_ids, BLOCK_SIZE, "train_ids")
    validate_sequence_length(val_ids, BLOCK_SIZE, "val_ids")

    # 5. sample one train batch
    x, y = get_batch(
        data=train_ids,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        device=device,
    )

    print("\nTrain batch check:")
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")

    print("\nFirst sample in train batch:")
    print("x[0]:", x[0])
    print("y[0]:", y[0])

    print("\nDecoded first sample:")
    print("x_text:", repr(tokenizer.decode(x[0].tolist())))
    print("y_text:", repr(tokenizer.decode(y[0].tolist())))

    # 6. sample one val batch
    x_val, y_val = get_batch(
        data=val_ids,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        device=device,
    )

    print("\nVal batch check:")
    print(f"x_val shape: {x_val.shape}")
    print(f"y_val shape: {y_val.shape}")


if __name__ == "__main__":
    main()