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

**Fix:** Added a minimum-length check in `_sample`:
```python
if len(generated) == 0:
    return " "   # fallback — score will be 0, which is valid
```

**Key lesson:** RL loops fail silently — the loss becomes NaN and training appears to continue. Always add `assert not torch.isnan(loss)` at each step during development.
