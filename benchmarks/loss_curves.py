"""Plot training and SFT loss curves from saved checkpoints.
Run: uv run python benchmarks/loss_curves.py
"""
import os, torch, matplotlib.pyplot as plt

def plot_loss(ckpt_path: str, label: str, key: str = "train"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    log  = ckpt.get("loss_log", [])
    steps = [e["step"] for e in log]
    vals  = [e.get(key, e.get("loss", 0)) for e in log]
    plt.plot(steps, vals, label=label)

if __name__ == "__main__":
    os.makedirs("benchmarks/plots", exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Pretraining Loss (TinyShakespeare)")
    if os.path.exists("checkpoints/tiny_pretrain.pt"):
        plot_loss("checkpoints/tiny_pretrain.pt", "train loss", "train")
        plot_loss("checkpoints/tiny_pretrain.pt", "val loss",   "val")
    plt.xlabel("step"); plt.ylabel("loss"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("SFT Loss (Alpaca, GPT-2 small)")
    if os.path.exists("checkpoints/sft.pt"):
        plot_loss("checkpoints/sft.pt", "SFT loss", "loss")
    plt.xlabel("step"); plt.ylabel("masked loss"); plt.legend()

    plt.tight_layout()
    plt.savefig("benchmarks/plots/loss_curves.png", dpi=150)
    print("Saved: benchmarks/plots/loss_curves.png")
