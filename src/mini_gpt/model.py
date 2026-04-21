import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadSelfAttention(nn.Module):
    def __init__(self, n_embd, block_size, dropout=0.1):
        super().__init__()

        self.n_embd = n_embd
        self.block_size = block_size

        # project input x to Q, K, V
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)

        self.dropout = nn.Dropout(dropout)

        # causal mask: only allow attending to current and previous positions
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask)

    def forward(self, x):
        """
        x: [B, T, C]
        return: [B, T, C]
        """
        B, T, C = x.shape

        # 1) compute Q, K, V
        # q, k, v: [B, T, C]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # 2) attention scores
        # k.transpose(-2, -1): [B, C, T]
        # att: [B, T, T]
        att = q @ k.transpose(-2, -1)

        # 3) scale
        att = att / math.sqrt(C)

        # 4) causal mask
        # self.mask[:T, :T]: [T, T]
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        # 5) softmax -> attention weights
        # att_weights: [B, T, T]
        att_weights = F.softmax(att, dim=-1)
        att_weights = self.dropout(att_weights)

        # 6) weighted sum of values
        # out: [B, T, C]
        out = att_weights @ v

        return out


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.attn = SingleHeadSelfAttention(
            n_embd=n_embd,
            block_size=block_size,
            dropout=dropout,
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

        # self-attention
        # x: [B, T, C]
        x = self.attn(x)

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