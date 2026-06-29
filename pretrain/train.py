"""
Tiny pretraining on TinyShakespeare.
Run: uv run python pretrain/train.py
"""
import math, os, time
import torch
import torch.nn.functional as F
from model.config import TINY_CONFIG
from model.gpt import GPT
from pretrain.data import get_batch

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE       = 8
GRAD_ACCUM_STEPS = 4        # effective batch = 32; simulates multi-GPU on single M2 Pro
MAX_STEPS        = 5_000
WARMUP_STEPS     = 200
MAX_LR           = 3e-4
MIN_LR           = 3e-5
EVAL_INTERVAL    = 200
CHECKPOINT_DIR   = "checkpoints"
DEVICE           = "mps" if torch.backends.mps.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────

def get_lr(step: int) -> float:
    # Linear warmup then cosine decay — standard for transformer pretraining
    if step < WARMUP_STEPS:
        return MAX_LR * step / WARMUP_STEPS
    if step > MAX_STEPS:
        return MIN_LR
    ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)

@torch.no_grad()
def estimate_loss(model: GPT, eval_iters: int = 50) -> dict[str, float]:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(split, TINY_CONFIG.block_size, BATCH_SIZE, DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model = GPT(TINY_CONFIG).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params | device: {DEVICE}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=MAX_LR, betas=(0.9, 0.95), weight_decay=0.1
    )

    loss_log = []
    t0 = time.time()

    for step in range(MAX_STEPS):
        # Update learning rate every step (cosine schedule)
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation: sum gradients over micro-batches before stepping.
        # Equivalent to training with batch_size * grad_accum_steps samples.
        optimizer.zero_grad(set_to_none=True)
        for _ in range(GRAD_ACCUM_STEPS):
            x, y = get_batch("train", TINY_CONFIG.block_size, BATCH_SIZE, DEVICE)
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            ) / GRAD_ACCUM_STEPS
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
            losses = estimate_loss(model)
            dt = time.time() - t0
            print(f"step {step:5d} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | lr {lr:.2e} | {dt:.1f}s")
            loss_log.append({"step": step, **losses})
            t0 = time.time()

    ckpt_path = os.path.join(CHECKPOINT_DIR, "tiny_pretrain.pt")
    torch.save({"model": model.state_dict(), "config": TINY_CONFIG, "loss_log": loss_log},
               ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")

if __name__ == "__main__":
    train()
