# %% [markdown]
# # 06 - Reinforcement Learning
# Shows reward curve, KL divergence, and a reward-hacking example.

# %%
import torch, matplotlib.pyplot as plt

try:
    ckpt = torch.load("../checkpoints/rl.pt", map_location="cpu", weights_only=False)
    log  = ckpt["reward_log"]
    steps   = [e["step"]   for e in log]
    rewards = [e["reward"] for e in log]

    plt.figure(figsize=(8, 3))
    plt.plot(steps, rewards)
    plt.xlabel("step"); plt.ylabel("mean reward"); plt.title("RL Reward Curve")
    plt.axhline(y=0.5, color='r', linestyle='--', label='target')
    plt.legend()
    plt.savefig("../benchmarks/plots/rl_reward.png", dpi=120, bbox_inches="tight"); plt.show()
    print(f"Final mean reward: {rewards[-1]:.3f}")
except FileNotFoundError:
    print("Run rl/trainer.py first.")

# %% [markdown]
# ## Group-relative advantage - why we normalise rewards within the group

# %%
import numpy as np
rewards_raw = np.array([0.1, 0.0, 0.7, 0.5, 0.2, 0.9, 0.3, 0.6])
advantages  = (rewards_raw - rewards_raw.mean()) / (rewards_raw.std() + 1e-8)
print("Raw rewards:       ", rewards_raw.round(2))
print("Group advantages:  ", advantages.round(2))
print("Mean reward:       ", rewards_raw.mean().round(3))
# Advantages sum to ~0 - the model updates toward above-average completions
