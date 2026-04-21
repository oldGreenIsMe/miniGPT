import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd

        # token embedding: [vocab_size] -> [n_embd]
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # position embedding: [block_size] -> [n_embd]
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.ln_f = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

        # final projection to vocab logits
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

        # 1) token embeddings
        # idx: [B, T]
        # tok_emb: [B, T, C]
        tok_emb = self.token_embedding_table(idx)

        # 2) position embeddings
        # positions: [T]
        # pos_emb: [T, C]
        positions = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(positions)

        # 3) add token + position
        # tok_emb: [B, T, C]
        # pos_emb: [T, C]
        # x: [B, T, C]
        x = tok_emb + pos_emb

        x = self.dropout(x)
        x = self.ln_f(x)

        # 4) project to logits
        # logits: [B, T, vocab_size]
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape

            # reshape for cross_entropy
            # logits: [B*T, C]
            # targets: [B*T]
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss