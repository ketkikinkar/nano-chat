# CHALLENGES.md

Real bugs encountered, what caused them, how they were fixed, and the key lesson.
These are interview-ready explanations.

---

## Challenge 1: Off-by-one in SFT loss shift

**What broke:** SFT loss was unexpectedly high even on training data.

**Root cause:** `logits[:, :-1]` and `targets[:, 1:]` were not aligned — the logit at position `t` predicts position `t+1`, so both must be shifted.

**Fix:**
```python
logits  = logits[:, :-1, :]   # (B, T-1)
targets = input_ids[:, 1:]     # (B, T-1) ← same length, now aligned
mask    = loss_mask[:, 1:]     # (B, T-1) ← shifted to match targets
```

**Key lesson:** Always draw the index alignment on paper before coding the loss. One-off errors here cause silent incorrect gradients, not crashes.

---

## Challenge 2: `mask.sum()` vs `seq_len` in SFT denominator

**What broke:** SFT loss was very small for conversations with long prompts and short answers — the gradient signal was being diluted.

**Root cause:** Dividing by `seq_len` instead of `mask.sum()`. A 500-token prompt + 10-token answer means only 10 tokens contribute to the numerator, but the denominator was 510 — a 51× dilution.

**Fix:** `return (per_token * mask).sum() / mask.sum()`

**Key lesson:** The denominator of the loss controls the effective learning rate. An incorrect denominator is as bad as the wrong learning rate — and harder to detect.

---

## Challenge 3: GPT-2 weight loading — Conv1D transpose

**What broke:** Model generated nonsense after loading GPT-2 weights.

**Root cause:** HuggingFace GPT-2 uses `Conv1D` which stores weights as `(in_features, out_features)`. `nn.Linear` expects `(out_features, in_features)`. Loading without transposing produced corrupted projections.

**Fix:**
```python
needs_transpose = any(hf_key.endswith(t) for t in ["c_attn.weight", "c_proj.weight", "c_fc.weight"])
if needs_transpose:
    sd[our_key].copy_(hf_val.t())   # .t() transposes
```

**Key lesson:** Weight loading bugs produce plausible-looking (but wrong) outputs. Always validate by checking that the model's logits on a known input match the HuggingFace model's output before proceeding.

---

## Challenge 4: MPS JIT warmup

**What broke:** First benchmark run showed 10× slower throughput than subsequent runs.

**Root cause:** MPS (Metal Performance Shaders) JIT-compiles the computation graph on the first call. This one-time cost made the first measurement an outlier.

**Fix:** Always run at least one warm-up pass before timing:
```python
_ = model(dummy_input)   # warm up MPS JIT
# now start timing
```

**Key lesson:** Benchmark methodology matters as much as the implementation. Always warm up, run multiple iterations, and report mean ± std.

---

## Challenge 5: RL NaN loss — empty completion

**What broke:** RL training produced NaN loss after ~50 steps.

**Root cause:** The model occasionally generated a completion of length 0 (sampled the EOS token immediately). `get_log_probs` on an empty sequence returned 0, and the advantage computation produced NaN.

**Fix:** The real protection is the epsilon in the advantage normalization — it prevents NaN whether from zero-variance reward groups or from edge-case completions:
```python
# In advantage normalization — this epsilon is why empty completions don't crash
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
```

**Key lesson:** Always add epsilon to std; it prevents NaN from zero-variance groups AND from edge-case completions. RL loops fail silently — the loss becomes NaN and training appears to continue. Always add `assert not torch.isnan(loss)` at each step during development.

---

## Challenge 6: KV-cache off-by-one — wrong first token + wrong positional embeddings

**What broke:** `generate_cached` produced different tokens from `generate_naive` on 9 of 10 random seeds, despite passing at seed=42. The single-seed test had passed by coincidence (all 8 generated tokens happened to be the same value).

**Root cause:** Two interleaved bugs: (1) the first decode step fed the last prompt token as input instead of sampling a new token from the prefill logits — the first generated token was wrong; (2) positional embeddings during decode started at position 0 instead of position `T_prompt + step`, so every generated token used the wrong position encoding.

**Fix:**
```python
# Prefill: run the full prompt once
logits_prefill = model(ids, pos_offset=0)
# Sample the FIRST new token from the last prefill position
first_token = sample(logits_prefill[0, -1])

# Patch model.forward to accept pos_offset for correct positions during decode
# In decode loop: pos_offset = T_prompt + step - 1
pos = torch.arange(pos_offset, pos_offset + T, device=device)

# Multi-seed parity test to prevent recurrence
for seed in [0, 1, 2]:
    torch.manual_seed(seed)
    naive = generate_naive(model, ids, max_new=8)
    torch.manual_seed(seed)
    cached = generate_cached(model, ids, max_new=8)
    assert torch.equal(naive, cached), f"Parity failed at seed {seed}"
```

**Key lesson:** Test with multiple random seeds — a bug that produces the right answer at seed=42 can fail on all others. Positional embedding bugs in KV-cache are especially insidious because the model still generates *something* plausible; only exact output parity reveals the error.
