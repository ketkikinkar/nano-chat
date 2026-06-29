"""Naive and KV-cache generation side by side.

Naive generation is O(T²): every step re-computes attention over all prior tokens.
KV-cache generation is O(T): each step only processes the new token; prior K/V
matrices are cached and reused.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch import Tensor
from model.gpt import GPT


def _sample_token(logits: Tensor, temperature: float, top_k: int) -> Tensor:
    """Sample or argmax from logits. temperature=0 → greedy (deterministic)."""
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[..., -1:]] = float('-inf')
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_naive(
    model: GPT,
    prompt_ids: Tensor,
    max_new: int,
    temperature: float = 1.0,
    top_k: int = 0,
) -> Tensor:
    """O(T²) generation: recomputes all prior tokens at every step.

    Simple and correct, but slow for long sequences because the attention matrix
    grows with every token generated.
    """
    ids = prompt_ids.unsqueeze(0)   # (1, T)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new):
            if ids.shape[1] >= model.config.block_size:
                break
            logits = model(ids)                              # (1, T, vocab)
            next_tok = _sample_token(logits[0, -1], temperature, top_k)  # (1,)
            ids = torch.cat([ids, next_tok.unsqueeze(0)], dim=1)
    return ids[0]


def generate_cached(
    model: GPT,
    prompt_ids: Tensor,
    max_new: int,
    temperature: float = 1.0,
    top_k: int = 0,
) -> Tensor:
    """O(T) generation with a KV-cache.

    The K and V tensors for all prior tokens are stored and reused.
    Each new step only runs attention on the single new token.

    KVCache structure: list of [K, V] per layer,
    each tensor shape (1, n_head, seq_so_far, head_dim).

    Positional embeddings must be offset during decode so that the new token's
    position matches what naive generation would compute — otherwise logits differ.
    """
    model.eval()

    # Monkey-patch the attention layers to accumulate K/V into a cache.
    # Each layer gets its own slot in this list.
    kv_cache: list[list[Tensor]] = [[] for _ in model.blocks]

    def _make_cached_forward(block_idx: int):
        """Return a forward function that appends K/V to the layer's cache slot."""

        def cached_forward(x: Tensor) -> Tensor:
            B, T, C = x.shape
            attn = model.blocks[block_idx].attn
            n_head, head_dim = attn.n_head, attn.head_dim

            qkv = attn.c_attn(x)
            q, k, v = qkv.split(attn.n_embd, dim=2)
            q = q.view(B, T, n_head, head_dim).transpose(1, 2)  # (B, nh, T, hd)
            k = k.view(B, T, n_head, head_dim).transpose(1, 2)
            v = v.view(B, T, n_head, head_dim).transpose(1, 2)

            # Append new K/V slices to the running cache
            cache = kv_cache[block_idx]
            if cache:
                k_full = torch.cat([cache[0], k], dim=2)
                v_full = torch.cat([cache[1], v], dim=2)
            else:
                k_full, v_full = k, v
            kv_cache[block_idx] = [k_full, v_full]

            # q attends over the full cached sequence.
            # No causal mask is needed during single-token decode because q is
            # always the *last* position — it should attend to all prior tokens.
            scale = math.sqrt(head_dim)
            scores = (q @ k_full.transpose(-2, -1)) / scale  # (B, nh, T, seq_so_far)

            # During prefill (T > 1) apply the causal mask so the prefill pass
            # produces identical outputs to naive's full-sequence pass.
            if T > 1:
                S = k_full.shape[2]
                scores = scores.masked_fill(
                    attn.mask[:, :, :T, :S] == 0, float('-inf')
                )

            weights = torch.softmax(scores, dim=-1)
            # attn_drop is a no-op at eval(); omitting it keeps the code clean
            out = (weights @ v_full).transpose(1, 2).contiguous().view(B, T, C)
            return attn.resid_drop(attn.c_proj(out))

        return cached_forward

    # Install cached forward for each attention layer
    for i in range(len(model.blocks)):
        model.blocks[i].attn.forward = _make_cached_forward(i)

    # Patch GPT.forward to support a pos_offset keyword argument so that
    # single-token decode steps use the correct absolute position index.
    # Without this, decoding token at position T_prompt would receive pos=[0]
    # instead of pos=[T_prompt], making logits differ from naive generation.
    orig_gpt_forward = model.forward.__func__  # unbound method

    def _patched_gpt_forward(self_model, idx: Tensor, pos_offset: int = 0) -> Tensor:
        B, T = idx.shape
        assert T + pos_offset <= self_model.config.block_size
        device = idx.device
        # Shift positional indices by offset — critical for decode parity
        pos = torch.arange(pos_offset, pos_offset + T, device=device)
        x = self_model.drop(self_model.wte(idx) + self_model.wpe(pos))
        for block in self_model.blocks:
            x = block(x)
        x = self_model.ln_f(x)
        return self_model.lm_head(x)

    model.forward = lambda idx, pos_offset=0: _patched_gpt_forward(model, idx, pos_offset)

    ids = prompt_ids.unsqueeze(0)   # (1, T_prompt)
    T_prompt = ids.shape[1]

    with torch.no_grad():
        # Prefill: process the entire prompt once to populate the KV caches.
        # Capture the logits so we can sample the FIRST new token from the
        # prefill output — no extra model call needed for that first step.
        logits_prefill = model(ids, pos_offset=0)   # (1, T_prompt, vocab)

        generated = []
        # Step 0: sample the first new token from prefill logits (position T_prompt-1).
        # The prefill pass already processed the last prompt token, so we must NOT
        # re-feed it; instead we use the logit it produced directly.
        next_tok = _sample_token(logits_prefill[0, -1], temperature, top_k)  # (1,)
        generated.append(next_tok.item())
        last_tok = next_tok.unsqueeze(0)   # (1, 1) — first generated token

        for step in range(1, max_new):
            # `last_tok` is the token at absolute position T_prompt + step - 1.
            # We have already generated `step` tokens (stored in `generated`).
            pos = T_prompt + step - 1
            if pos >= model.config.block_size:
                break
            # Feed last_tok at its correct absolute position.
            logits = model(last_tok, pos_offset=pos)  # (1, 1, vocab)
            next_tok = _sample_token(logits[0, -1], temperature, top_k)  # (1,)
            generated.append(next_tok.item())
            last_tok = next_tok.unsqueeze(0)

    # Restore original forward methods — remove monkey-patches so the model is
    # left in the same state it was found in.
    for i in range(len(model.blocks)):
        del model.blocks[i].attn.forward   # reveals the class method underneath
    del model.forward                       # reveals the class method underneath

    return torch.cat([prompt_ids, torch.tensor(generated, dtype=torch.long)])
