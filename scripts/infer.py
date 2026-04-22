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
    DEVICE,
)
from mini_gpt.tokenizer import CharTokenizer
from mini_gpt.model import MiniGPT


def main():
    device = DEVICE if torch.cuda.is_available() else "cpu"

    tokenizer = CharTokenizer.load(VOCAB_PATH)
    vocab_size = tokenizer.vocab_size

    model = MiniGPT(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT,
    ).to(device)

    state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    prompts = [
        "To be",
        "The king",
        "If love",
        "O my",
    ]

    for prompt in prompts:
        idx = torch.tensor([tokenizer.encode(prompt)], device=device)

        out = model.generate(idx, max_new_tokens=150)
        text = tokenizer.decode(out[0].tolist())

        print("\n" + "=" * 60)
        print(f"Prompt: {prompt}")
        print("-" * 60)
        print(text)


if __name__ == "__main__":
    main()