import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout=0.1):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask)

        self.head_size = head_size

    def forward(self, x):
        """
        x: [B, T, C]
        return: [B, T, head_size]
        """
        B, T, C = x.shape

        # q, k, v: [B, T, head_size]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # attention scores: [B, T, T]
        att = q @ k.transpose(-2, -1)
        att = att / math.sqrt(self.head_size)

        # casual mask
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        # attention weights: [B, T, T]
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # weighted sum: [B, T, head_size]
        out = att @ v
        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()

        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )

        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

        self.n_head = n_head
        self.head_size = head_size

    def forward(self, x):
        """
        x: B, T, C
        each head output: [B, T, head_size]
        concat output: [B, T, C]
        """
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        x: [B, T, C]
        return: [B, T, C]
        """
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.sa = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout
        )

        self.ffwd = FeedForward(
            n_embd=n_embd,
            dropout=dropout
        )

    def forward(self, x):
        """
        x: [B, T, C]
        return: [B, T, C]
        """
        # pre-norm + residual
        x = x + self.sa(self.ln1(x))

        # pre-norm + residual
        x = x + self.ffwd(self.ln2(x))

        return x