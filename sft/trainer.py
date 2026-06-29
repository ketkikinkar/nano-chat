"""SFT training loop — the primary differentiator of this project."""
from __future__ import annotations
import os, math, json
import torch
import torch.nn.functional as F
from torch import Tensor
from model.gpt import GPT
from model.config import GPT2_CONFIG, GPTConfig
from sft.data import build_batch
import tiktoken

# ── Hyperparameters ────────────────────────────────────────────────────────────
LR            = 2e-5
MAX_EPOCHS    = 3
BATCH_SIZE    = 4
GRAD_ACCUM    = 4
CHECKPOINT_DIR = "checkpoints"
DEVICE        = "mps" if torch.backends.mps.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────


def sft_loss(model: GPT, input_ids: Tensor, loss_mask: Tensor) -> Tensor:
    """Compute cross-entropy loss averaged over ASSISTANT tokens only.

    Dividing by mask.sum() (not seq_len) ensures long prompts don't dilute
    the gradient signal — the loss is always the average over assistant tokens
    regardless of how much prompt precedes them.
    """
    B, T = input_ids.shape
    vocab_size = model.config.vocab_size

    logits  = model(input_ids)             # (B, T, vocab_size)
    logits  = logits[:, :-1, :].contiguous()   # (B, T-1, vocab_size) — drop last
    targets = input_ids[:, 1:].contiguous()    # (B, T-1) — shift left
    mask    = loss_mask[:, 1:].contiguous()    # (B, T-1) — align with targets

    per_token_loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
        reduction="none",
    ).view(B, T - 1)                           # (B, T-1)

    # Guard: if mask is all zeros, return 0 instead of NaN
    denom = mask.sum()
    if denom == 0:
        return torch.tensor(0.0, device=input_ids.device, requires_grad=True)

    return (per_token_loss * mask).sum() / denom


def train(data_path: str = "data/alpaca.json"):
    """Fine-tune GPT-2 on Alpaca conversations using the SFT loss."""
    enc = tiktoken.get_encoding("gpt2")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    with open(data_path) as f:
        raw = json.load(f)

    # Format Alpaca entries as two-turn conversations
    def to_turns(item: dict) -> list[dict]:
        prompt = item["instruction"]
        if item.get("input"):
            prompt += f"\n\n{item['input']}"
        return [
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": item["output"]},
        ]

    conversations = [to_turns(item) for item in raw]
    print(f"Loaded {len(conversations)} conversations")

    model = GPT.from_pretrained("gpt2").to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95))

    loss_log = []
    step = 0

    for epoch in range(MAX_EPOCHS):
        import random; random.shuffle(conversations)
        for i in range(0, len(conversations), BATCH_SIZE):
            batch_convos = conversations[i : i + BATCH_SIZE]
            result = build_batch(batch_convos, enc, GPT2_CONFIG.block_size)
            if result is None:
                continue
            input_ids, loss_mask = result
            input_ids  = input_ids.to(DEVICE)
            loss_mask  = loss_mask.to(DEVICE)

            # set_to_none=True frees the gradient tensor memory rather than
            # zeroing it, which is faster and avoids a superfluous write
            optimizer.zero_grad(set_to_none=True)
            for _ in range(GRAD_ACCUM):
                loss = sft_loss(model, input_ids, loss_mask) / GRAD_ACCUM
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 100 == 0:
                print(f"epoch {epoch} step {step} | loss {loss.item() * GRAD_ACCUM:.4f}")
                loss_log.append({"step": step, "loss": loss.item() * GRAD_ACCUM})
            step += 1

    ckpt = {"model": model.state_dict(), "config": GPT2_CONFIG, "loss_log": loss_log}
    torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "sft.pt"))
    print(f"SFT checkpoint saved.")


if __name__ == "__main__":
    train()
