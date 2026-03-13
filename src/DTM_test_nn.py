#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:37:51 2026

@author: bipin
"""
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

    def split_heads(self, x):
        # x: (batch, seq_len, d_model) → (batch, n_heads, seq_len, d_k)
        batch, seq_len, _ = x.size()
        x = x.view(batch, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, x, mask=None):
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        attn = self.scaled_dot_product_attention(Q, K, V, mask)

        # Merge heads back: (batch, n_heads, seq_len, d_k) → (batch, seq_len, d_model)
        batch, _, seq_len, _ = attn.size()
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.W_o(attn)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class Transformer7D(nn.Module):
    """
    A simple transformer that takes vectors of length 7
    and generates vectors of length 7.

    Args:
        input_dim:  Size of input/output vectors (7).
        d_model:    Internal model dimension.
        n_heads:    Number of attention heads (must divide d_model).
        n_layers:   Number of stacked transformer blocks.
        d_ff:       Feed-forward hidden dimension.
        dropout:    Dropout probability.
    """
    def __init__(
        self,
        input_dim: int = 7,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project input_dim → d_model
        self.embedding = nn.Linear(input_dim, d_model)

        # Stack of transformer blocks
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # Project d_model → input_dim (back to 7)
        self.output_projection = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Args:
            x:    Tensor of shape (batch_size, seq_len, 7)
            mask: Optional attention mask

        Returns:
            Tensor of shape (batch_size, seq_len, 7)
        """
        x = self.embedding(x)           # (batch, seq_len, d_model)

        for layer in self.layers:
            x = layer(x, mask)          # (batch, seq_len, d_model)

        return self.output_projection(x)  # (batch, seq_len, 7)


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    
    model = Transformer7D(
        input_dim=6,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
    )

    # Batch of 3 sequences, each with 5 time steps of 7-dim vectors
    batch_size, seq_len, input_dim = 3, 5, 6
    x = torch.randn(batch_size, seq_len, input_dim)

    model.eval()
    with torch.no_grad():
        output = model(x)

   
