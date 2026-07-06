"""REINFORCE-style RL with group-relative advantage and KL penalty."""
from __future__ import annotations
import os
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor
from model.gpt import GPT
from model.config import TINY_CONFIG
from rl.reward import compute_reward
import tiktoken

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class RLConfig:
    G:              int   = 8      # completions per prompt
    kl_coeff:       float = 0.1    # weight on KL penalty - prevents reward hacking
    lr:             float = 1e-5
    max_steps:      int   = 500
    max_new_tokens: int   = 64


def get_log_probs(model: GPT, token_ids: Tensor) -> Tensor:
    """Sum of log probabilities of all tokens under the model.

    Used both for the policy gradient loss and the KL penalty.
    token_ids is a 1-D sequence; we shift by one so each token
    predicts the next - standard causal LM teacher-forcing.
    """
    ids = token_ids.unsqueeze(0)      # (1, T)
    logits = model(ids)               # (1, T, vocab)
    logits  = logits[:, :-1, :]      # (1, T-1, vocab)
    targets = ids[:, 1:]              # (1, T-1)
    log_p = F.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
    )
    return -log_p.sum()               # sum of log probs (negative cross-entropy)


def _sample(model: GPT, prompt: Tensor, max_new: int, enc: tiktoken.Encoding) -> str:
    """Autoregressive sampling with temperature=0.8.

    Temperature < 1 sharpens the distribution, trading diversity for
    coherence - important so the reward signal doesn't collapse to noise.
    """
    model.eval()
    with torch.no_grad():
        ids = prompt.unsqueeze(0).to(DEVICE)
        for _ in range(max_new):
            if ids.shape[1] >= model.config.block_size:
                break
            logits = model(ids)
            next_tok = torch.multinomial(
                torch.softmax(logits[0, -1] / 0.8, dim=-1), 1
            )
            ids = torch.cat([ids, next_tok.unsqueeze(0)], dim=1)
    generated = ids[0, prompt.shape[0]:].tolist()
    return enc.decode(generated)


def rl_step(
    model: GPT,
    ref_model: GPT,
    prompt_tokens: Tensor,
    config: RLConfig,
    enc: tiktoken.Encoding,
) -> Tensor:
    """One REINFORCE update step with group-relative advantage and KL penalty.

    Group-relative advantage: normalising rewards within the group (r - mean)/std
    gives a stable training signal regardless of absolute reward magnitude.
    The KL penalty anchors the RL model to the SFT reference to prevent
    degenerate reward-hacking completions.
    """
    model.train()

    # 1. Sample G completions from the current policy
    completions = [
        _sample(model, prompt_tokens, config.max_new_tokens, enc)
        for _ in range(config.G)
    ]

    # 2. Score and compute group-relative advantage
    rewards = torch.tensor([compute_reward(c) for c in completions], dtype=torch.float32)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # 3. Policy gradient loss (REINFORCE)
    # We accumulate by summation and divide by G to get the mean
    pg_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
    enc_completions = [
        torch.tensor(enc.encode_ordinary(c), dtype=torch.long).to(DEVICE)
        for c in completions
    ]
    for completion_ids, adv in zip(enc_completions, advantages):
        full_ids = torch.cat([prompt_tokens.to(DEVICE), completion_ids])
        lp = get_log_probs(model, full_ids)
        # Negative advantage * log_prob: maximise reward via gradient ascent
        pg_loss = pg_loss + (-adv.to(DEVICE) * lp)
    pg_loss = pg_loss / config.G

    # 4. KL penalty: keeps RL model close to the SFT checkpoint.
    # Without it, the model can find degenerate completions that score high but are nonsense.
    # ref_model must be frozen (requires_grad=False) - we only back-prop through pol_lp.
    ref_model.eval()
    # KL computed on the first completion only - cheap approximation for a demo
    full_ids = torch.cat([prompt_tokens.to(DEVICE), enc_completions[0]])
    with torch.no_grad():
        ref_lp = get_log_probs(ref_model, full_ids)
    pol_lp = get_log_probs(model, full_ids)
    kl = pol_lp - ref_lp.detach()   # positive when model diverges from reference

    return pg_loss + config.kl_coeff * kl


def train(sft_ckpt_path: str = "checkpoints/tiny_pretrain.pt"):
    enc = tiktoken.get_encoding("gpt2")
    os.makedirs("checkpoints", exist_ok=True)

    ckpt = torch.load(sft_ckpt_path, map_location=DEVICE)
    model = GPT(TINY_CONFIG).to(DEVICE)
    model.load_state_dict(ckpt["model"])

    # Frozen reference: anchors the RL model to the SFT checkpoint via KL penalty.
    # requires_grad=False ensures no gradients flow into the reference copy.
    ref_model = GPT(TINY_CONFIG).to(DEVICE)
    ref_model.load_state_dict(ckpt["model"])
    for p in ref_model.parameters():
        p.requires_grad = False

    rl_cfg = RLConfig()
    optimizer = torch.optim.AdamW(model.parameters(), lr=rl_cfg.lr)

    # Fixed prompts from TinyShakespeare
    prompts = [
        "To be or not to be",
        "All the world's a stage",
        "What a piece of work is man",
    ]
    prompt_ids = [
        torch.tensor(enc.encode_ordinary(p), dtype=torch.long) for p in prompts
    ]

    reward_log = []
    for step in range(rl_cfg.max_steps):
        prompt = prompt_ids[step % len(prompt_ids)]
        # set_to_none=True avoids zeroing gradients - frees memory instead
        optimizer.zero_grad(set_to_none=True)
        loss = rl_step(model, ref_model, prompt, rl_cfg, enc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0:
            # Quick reward estimate on 4 samples for logging
            completions = [_sample(model, prompt, 32, enc) for _ in range(4)]
            mean_r = sum(compute_reward(c) for c in completions) / 4
            print(f"step {step:4d} | loss {loss.item():.4f} | reward {mean_r:.3f}")
            reward_log.append({"step": step, "reward": mean_r})

    torch.save(
        {"model": model.state_dict(), "config": TINY_CONFIG, "reward_log": reward_log},
        "checkpoints/rl.pt",
    )
    print("RL checkpoint saved.")


if __name__ == "__main__":
    train()
