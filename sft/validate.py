"""
Validate that our SFT loss matches the nanochat-mlx reference within tolerance.
Run: uv run python sft/validate.py

Requires:
  - checkpoints/sft.pt (from sft/trainer.py)
  - The nanochat-mlx repo cloned to /tmp/nanochat-mlx
"""
import torch
import json

TOLERANCE = 5e-3   # loss curves must be within this at each step

def load_our_loss_log() -> list[dict]:
    ckpt = torch.load("checkpoints/sft.pt", map_location="cpu")
    return ckpt["loss_log"]

def load_reference_loss_log(path: str = "/tmp/nanochat-mlx/loss_log.json") -> list[dict]:
    """Load the reference loss log saved by nanochat-mlx's SFT run."""
    with open(path) as f:
        return json.load(f)

def compare(our_log: list[dict], ref_log: list[dict]):
    print(f"\n{'Step':>6}  {'Our Loss':>10}  {'Ref Loss':>10}  {'Delta':>8}  {'Status':>6}")
    print("-" * 55)
    passed = True
    for ours, ref in zip(our_log[:20], ref_log[:20]):
        delta = abs(ours["loss"] - ref["loss"])
        ok = delta < TOLERANCE
        passed = passed and ok
        status = "OK" if ok else "FAIL"
        print(f"{ours['step']:>6}  {ours['loss']:>10.4f}  {ref['loss']:>10.4f}  {delta:>8.4f}  {status:>6}")
    print()
    if passed:
        print("✓ SFT loss curves match reference within tolerance.")
    else:
        print("✗ Curves diverged — check mask, averaging, or optimizer.")
    return passed

if __name__ == "__main__":
    our  = load_our_loss_log()
    ref  = load_reference_loss_log()
    compare(our, ref)
