# %% [markdown]
# # 03 — Transformer Architecture
# Builds the full block and verifies residual connections.

# %%
import torch
from model.config import GPTConfig
from model.blocks import TransformerBlock

CFG = GPTConfig(n_layer=2, n_head=4, n_embd=64, block_size=16, vocab_size=100)
block = TransformerBlock(CFG)

# %% [markdown]
# ## Residual connections — the gradient highway

# %%
torch.manual_seed(0)
x = torch.randn(1, 8, 64)
out = block(x)
print(f"Input  std:  {x.std().item():.3f}")
print(f"Output std:  {out.std().item():.3f}")
# If residuals work, output scale should be similar to input scale

# %% [markdown]
# ## Parameter count breakdown

# %%
from model.gpt import GPT
from model.config import TINY_CONFIG, GPT2_CONFIG

for name, cfg in [("Tiny (~20M)", TINY_CONFIG), ("GPT-2 small (124M)", GPT2_CONFIG)]:
    model = GPT(cfg)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n{name}: {total/1e6:.1f}M total params")
    for n, m in model.named_children():
        p = sum(x.numel() for x in m.parameters())
        print(f"  {n:<12} {p/1e6:.2f}M")
