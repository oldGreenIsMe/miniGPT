import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_gpt.modules import Block


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd=n_embd,
                    n_head=n_head,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        idx: [B, T]
        targets: [B, T] or None
        """
        B, T = idx.shape

        if T > self.block_size:
            raise ValueError(
                f"Sequence length T={T} exceeds block_size={self.block_size}"
            )

        # token embeddings: [B, T, C]
        tok_emb = self.token_embedding_table(idx)

        # position embeddings: [T, C]
        positions = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(positions)

        # x: [B, T, C]
        x = tok_emb + pos_emb
        x = self.dropout(x)

        # transformer blocks
        x = self.blocks(x)

        # final norm
        x = self.ln_f(x)

        # logits: [B, T, vocab_size]
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        idx: [B, T_start]
        return: [B, T_start + max_new_tokens]
        """
        self.eval()

        for _ in range(max_new_tokens):
            # 1) only keep the last block_size tokens
            idx_cond = idx[:, -self.block_size:]

            # 2) forward
            logits, _ = self(idx_cond)

            # 3) take the logits at the last time step
            # logits: [B, T, vocab_size]
            # last_logits: [B, vocab_size]
            last_logits = logits[:, -1, :]

            # 4) convert to probabilities
            probs = F.softmax(last_logits, dim=-1)

            # 5) sample next token
            # next_token: [B, 1]
            next_token = torch.multinomial(probs, num_samples=1)

            # 6) append to sequence
            idx = torch.cat((idx, next_token), dim=1)

        return idx