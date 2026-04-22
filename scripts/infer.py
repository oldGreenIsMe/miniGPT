import argparse
import random
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

SRC_ROOT = PROJECT_ROOT / "src"
sys.path.append(str(SRC_ROOT))

from configs.base_config import (
    VOCAB_PATH,
    BEST_MODEL_PATH,
    BLOCK_SIZE,
    N_EMBD,
    N_HEAD,
    N_LAYER,
    DROPOUT,
    SEED,
    DEVICE,
)
from mini_gpt.tokenizer import CharTokenizer
from mini_gpt.model import MiniGPT


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="To be")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    args = parser.parse_args()

    set_seed(SEED)

    device = DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. load tokenizer
    tokenizer = CharTokenizer.load(VOCAB_PATH)
    vocab_size = tokenizer.vocab_size
    print(f"Loaded vocab size: {vocab_size}")

    # 2. build model
    model = MiniGPT(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT,
    ).to(device)

    # 3. load trained weights
    state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from: {BEST_MODEL_PATH}")

    # 4. encode prompt
    start_text = args.start
    start_ids = tokenizer.encode(start_text)
    idx = torch.tensor([start_ids], dtype=torch.long, device=device)  # [1, T]

    print(f"\nPrompt: {repr(start_text)}")
    print(f"Prompt ids shape: {idx.shape}")
    print(f"temperature: {args.temperature}")
    print(f"top_k: {args.top_k}")

    # 5. generate
    out_idx = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    # 6. decode
    out_text = tokenizer.decode(out_idx[0].tolist())

    print("\nGenerated text:")
    print("-" * 50)
    print(out_text)
    print("-" * 50)


if __name__ == "__main__":
    main()