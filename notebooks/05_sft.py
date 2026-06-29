# %% [markdown]
# # 05 — Supervised Fine-Tuning
# Shows the loss mask side-by-side with tokens. This is the differentiator.

# %%
import torch
import tiktoken
from sft.data import format_conversation, build_batch

enc = tiktoken.get_encoding("gpt2")

# %% [markdown]
# ## Visualise the loss mask — the most important concept in SFT

# %%
convo = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "What is 2+2?"},
    {"role": "assistant", "content": "It is 4."},
]

ids, mask = build_batch([convo], enc, block_size=128)
ids, mask = ids[0].tolist(), mask[0].tolist()

print(f"{'Token':>12}  {'ID':>6}  {'Mask':>6}  {'Trains?'}")
print("-" * 45)
for i, (tok_id, m) in enumerate(zip(ids[:60], mask[:60])):
    if tok_id == 0: break
    tok = repr(enc.decode([tok_id]))
    trains = "YES ←" if m == 1.0 else ""
    print(f"{tok:>12}  {tok_id:>6}  {m:>6.1f}  {trains}")

# %% [markdown]
# ## Before vs after masking — loss comparison

# %%
from model.config import GPT2_CONFIG
from model.gpt import GPT
from sft.trainer import sft_loss

try:
    ckpt  = torch.load("../checkpoints/sft.pt", map_location="cpu", weights_only=False)
    model = GPT(GPT2_CONFIG)
    model.load_state_dict(ckpt["model"])
    model.eval()

    input_ids, loss_mask = build_batch([convo], enc, block_size=GPT2_CONFIG.block_size)
    full_mask   = torch.ones_like(loss_mask)
    loss_masked = sft_loss(model, input_ids, loss_mask)
    loss_full   = sft_loss(model, input_ids, full_mask)
    print(f"\nLoss (assistant tokens only): {loss_masked.item():.4f}")
    print(f"Loss (all tokens):            {loss_full.item():.4f}")
except FileNotFoundError:
    print("Run sft/trainer.py first to generate checkpoints/sft.pt")
