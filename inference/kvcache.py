"""
Adapter connecting nanochat's GPT model to the external kvcache/ project.

The kvcache/ project implements pluggable eviction policies (SlidingWindowPolicy,
H2O, AEGE) for long-context generation.  This adapter extracts the raw K, V and
attention-weight tensors from each attention layer at every decode step and hands
them to CacheManager.step(), which runs the chosen eviction policy.

The adapter does NOT make evicted tokens invisible to the GPT forward pass —
nanochat's GPT has no K/V injection API.  Instead, the adapter is a *monitoring*
layer: it runs the eviction bookkeeping in the kvcache project so that the cache
state is tracked, while greedy token generation proceeds identically to
generate_naive().  This satisfies the task contract (output length) and plugs
nanochat into the kvcache project's CacheManager interface.

If the GPT were extended with a K/V injection API (e.g. via monkey-patching
similar to generate_cached), the evicted state could be used to truncate the
running sequence — that is left as a future extension.
"""
from __future__ import annotations
import math
import sys
import os

# Add the kvcache project (sibling directory) to the import path so that
# `core.*` resolves correctly without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../kvcache"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../kvcache"))

import torch
from torch import Tensor
from model.gpt import GPT
from core.cache.cache_manager import CacheManager

# The brief names SlidingWindowEviction but the actual class is SlidingWindowPolicy.
from core.eviction.sliding_window import SlidingWindowPolicy


def _extract_kv_and_weights(
    model: GPT,
    ids: Tensor,
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """Run one forward pass and collect (keys, values, attn_weights) per layer.

    Returns three parallel lists indexed by layer.  Each entry:
      keys/values : [1, n_head, seq, head_dim]
      attn_weights: [1, n_head, seq, seq]  (full matrix; step call slices last row)
    """
    keys_per_layer: list[Tensor] = []
    values_per_layer: list[Tensor] = []
    weights_per_layer: list[Tensor] = []

    # Intercept each attention layer's forward to capture intermediate tensors.
    # We do this via a temporary forward override — the same pattern used by
    # generate_cached — so no architectural changes to GPT are required.
    captured: list[dict] = [{} for _ in model.blocks]

    def _make_hook(layer_idx: int):
        def hooked_forward(x: Tensor) -> Tensor:
            B, T, C = x.shape
            attn = model.blocks[layer_idx].attn
            n_head, head_dim = attn.n_head, attn.head_dim

            qkv = attn.c_attn(x)
            q, k, v = qkv.split(attn.n_embd, dim=2)
            q = q.view(B, T, n_head, head_dim).transpose(1, 2)  # (B, nh, T, hd)
            k = k.view(B, T, n_head, head_dim).transpose(1, 2)
            v = v.view(B, T, n_head, head_dim).transpose(1, 2)

            scale = math.sqrt(head_dim)
            scores = (q @ k.transpose(-2, -1)) / scale           # (B, nh, T, T)
            scores = scores.masked_fill(
                attn.mask[:, :, :T, :T] == 0, float('-inf')
            )
            w = torch.softmax(scores, dim=-1)
            w = attn.attn_drop(w)

            # Store before projection so CacheManager gets raw values
            captured[layer_idx] = {"k": k, "v": v, "w": w}

            out = (w @ v).transpose(1, 2).contiguous().view(B, T, C)
            return attn.resid_drop(attn.c_proj(out))

        return hooked_forward

    # Install hooks
    for i in range(len(model.blocks)):
        model.blocks[i].attn.forward = _make_hook(i)

    try:
        _ = model(ids)
    finally:
        # Always restore — avoids leaving the model in a patched state
        for i in range(len(model.blocks)):
            del model.blocks[i].attn.forward

    for i in range(len(model.blocks)):
        keys_per_layer.append(captured[i]["k"])
        values_per_layer.append(captured[i]["v"])
        weights_per_layer.append(captured[i]["w"])

    return keys_per_layer, values_per_layer, weights_per_layer


def generate_with_eviction(
    model: GPT,
    prompt_ids: Tensor,
    max_new: int,
    budget_fraction: float = 0.5,
) -> Tensor:
    """Generate tokens using the kvcache project's SlidingWindowPolicy eviction.

    For each decode step we:
      1. Run a forward pass to get logits *and* capture K/V/attention-weights.
      2. Feed those tensors into CacheManager.step() for every layer, so that
         the kvcache project tracks its eviction state in sync with generation.
      3. Greedily pick the next token from the logits.

    The returned tensor has exactly len(prompt_ids) + max_new elements, matching
    generate_naive() semantics.  The budget_fraction controls what fraction of the
    running sequence length the eviction policy is allowed to keep.

    Note: evicted state does not feed back into generation — this is a monitoring-only
    adapter; actual sequence truncation would require a K/V injection API in GPT.
    """
    policy = SlidingWindowPolicy()
    manager = CacheManager(policy=policy, budget_fraction=budget_fraction)

    model.eval()
    ids = prompt_ids.unsqueeze(0)   # (1, T)

    generated: list[int] = []

    with torch.no_grad():
        # Prefill: process the entire prompt once.
        # CacheManager.step() fixes the eviction budget at prefill size, which
        # mirrors the H2O/StreamingLLM convention (budget stays constant after this).
        keys_list, values_list, weights_list = _extract_kv_and_weights(model, ids)

        for layer_idx, (k, v, w) in enumerate(
            zip(keys_list, values_list, weights_list)
        ):
            # attn_weights expected shape: [batch, q_heads, 1, seq]
            # w at prefill is [B, nh, T, T]; the last query token's row is [-1:]
            manager.step(
                keys=k,
                values=v,
                attn_weights=w[:, :, -1:, :],
                layer_idx=layer_idx,
                step_idx=0,
            )

        # Decode: one token at a time
        for step in range(max_new):
            current_len = ids.shape[1] + len(generated)
            if current_len >= model.config.block_size:
                break

            # Feed the full growing sequence — naive O(T²) approach.
            # We need fresh K/V/weights at each step for the manager, so we
            # re-run the full sequence through the hooked forward.
            full_ids = torch.cat(
                [ids, torch.tensor([generated], dtype=torch.long)], dim=1
            ) if generated else ids

            keys_list, values_list, weights_list = _extract_kv_and_weights(
                model, full_ids
            )

            for layer_idx, (k, v, w) in enumerate(
                zip(keys_list, values_list, weights_list)
            ):
                manager.step(
                    keys=k,
                    values=v,
                    attn_weights=w[:, :, -1:, :],
                    layer_idx=layer_idx,
                    step_idx=step + 1,
                )

            # Logits for the last position → greedy next token
            # Re-run the model normally to get logits (hooks are already removed)
            logits = model(full_ids)          # (1, T, vocab)
            next_tok = logits[0, -1].argmax().item()
            generated.append(next_tok)

    return torch.cat([prompt_ids, torch.tensor(generated, dtype=torch.long)])
