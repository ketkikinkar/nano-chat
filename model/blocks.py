import torch
import torch.nn as nn
from model.config import GPTConfig
from model.attention import MultiHeadAttention


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # 4x expansion: stores combinations of features. Empirically optimal for GPT-class models.
        self.fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()   # smoother than ReLU; GPT-2 standard
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(self.gelu(self.fc(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: normalise BEFORE each sub-layer. More stable than Post-LN at init;
        # no careful warmup schedule required.
        x = x + self.attn(self.ln_1(x))   # attention sub-layer with residual
        x = x + self.mlp(self.ln_2(x))    # feed-forward sub-layer with residual
        return x
