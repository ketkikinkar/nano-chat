import math
import torch
import torch.nn as nn
from model.config import GPTConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Single projection for Q, K, V - one matmul instead of three (faster on MPS)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # Causal mask: lower-triangular 1s. Registered as buffer so it moves to
        # the right device with the model and is not treated as a learned parameter.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project and split into Q, K, V heads
        qkv = self.c_attn(x)                            # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)         # each (B, T, C)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # Scaling by sqrt(head_dim) keeps softmax from saturating when head_dim is large
        scale = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) / scale      # (B, nh, T, T)

        # Zero out future positions so the model cannot attend forward in time
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        out = weights @ v                                # (B, nh, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        return self.resid_drop(self.c_proj(out))
