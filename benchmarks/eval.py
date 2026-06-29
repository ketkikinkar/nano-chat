"""Evaluate base, SFT, and RL checkpoints on a simple perplexity metric.
Run: uv run python benchmarks/eval.py
"""
import torch, math
import torch.nn.functional as F
from model.gpt import GPT
from model.config import TINY_CONFIG
from pretrain.data import get_batch

DEVICE    = "mps" if torch.backends.mps.is_available() else "cpu"
EVAL_ITERS = 100

@torch.no_grad()
def eval_perplexity(model: GPT, split: str = "val") -> float:
    model.eval()
    losses = []
    for _ in range(EVAL_ITERS):
        x, y = get_batch(split, TINY_CONFIG.block_size, 4, DEVICE)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    return math.exp(sum(losses) / len(losses))

if __name__ == "__main__":
    print("\nEval — val-set perplexity per checkpoint (lower = better)")
    print("-" * 50)

    for name, path in [
        ("base (pretrain)",  "checkpoints/tiny_pretrain.pt"),
        ("RL",               "checkpoints/rl.pt"),
    ]:
        try:
            ckpt  = torch.load(path, map_location=DEVICE)
            model = GPT(TINY_CONFIG).to(DEVICE)
            model.load_state_dict(ckpt["model"])
            ppl = eval_perplexity(model)
            print(f"  {name:<25} perplexity = {ppl:.2f}")
        except FileNotFoundError:
            print(f"  {name:<25} (checkpoint not found — run training first)")
