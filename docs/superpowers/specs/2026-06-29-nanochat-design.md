# nanochat - Design Spec
**Date:** 2026-06-29
**Hardware:** Apple M2 Pro, 16 GB unified memory
**Framework:** PyTorch (MPS backend) primary · MLX comparison track
**Series position:** Project #2 of 8 (after bpe-tokenizer, before DPO)

---

## 1. Purpose

Build and understand the entire lifecycle of a ChatGPT-style assistant by:

1. Implementing the GPT transformer architecture from scratch in PyTorch
2. Running a tiny pretraining loop on TinyShakespeare to understand the training loop
3. Loading a GPT-2 checkpoint for the "real" SFT + inference tracks
4. Reimplementing the SFT training loop from scratch (the differentiator)
5. Implementing a basic REINFORCE-style RL loop
6. Integrating the existing KV-cache project into inference
7. Producing all documentation artifacts (notebooks, CHALLENGES.md, benchmarks, README)

**Resume framing:** "Implemented the GPT-2 architecture from scratch in PyTorch, trained it through the full pipeline (pretrain → SFT → RL → KV-cache inference), and reimplemented the SFT loop including conversation formatting and loss masking on assistant turns."

---

## 2. Scope

**In scope:**
- GPT-2-class transformer architecture implemented from scratch (attention, blocks, full model)
- Tiny pretraining on TinyShakespeare (~10M tokens) to validate the training loop
- GPT-2 small checkpoint (124M) for the SFT + inference tracks
- SFT loop reimplementation with ChatML format and loss masking
- Basic REINFORCE RL loop with rule-based reward and KL penalty
- KV-cache integration (referencing existing `kvcache/` project)
- Chat web UI via nanochat-mlx reference
- 7 Jupyter notebooks (one per stage), all runnable top-to-bottom
- Benchmarks: MPS vs MLX throughput, loss curves, eval scores
- CHALLENGES.md, RESUME_ADDITIONS.txt, README

**Out of scope:**
- Distributed training (torchrun, DDP) - single M2 Pro
- FineWeb full dataset pretraining - use TinyShakespeare or 10M-token slice
- Learned reward model - rule-based reward only
- Production serving, auth, scaling
- RLHF with PPO - REINFORCE-style only

**Stretch goals:**
- Implement FlashAttention manually and compare memory/speed
- Add MPS vs MLX parity test suite
- Swap RL reward and document behavior change

---

## 3. Two parallel training tracks

| Track | Purpose | Model config | Data |
|---|---|---|---|
| **Tiny from-scratch** | Understand the full loop | n_layer=6, n_head=6, n_embd=384 (~20M params) | TinyShakespeare |
| **GPT-2 checkpoint** | Portfolio-grade SFT validation | n_layer=12, n_head=12, n_embd=768, block_size=512 (124M) | Alpaca (52K conversations) |

Track 1 proves you understand every component. Track 2 is what goes in the README benchmark table.

---

## 4. Project structure

```
nanochat/
├── model/
│   ├── config.py          # GPTConfig dataclass
│   ├── attention.py       # MultiHeadAttention
│   ├── blocks.py          # TransformerBlock (attn + MLP + norms)
│   └── gpt.py             # Full GPT model + weight tying
├── pretrain/
│   ├── data.py            # Tokenise, memmap, batch sampler
│   └── train.py           # Training loop (MPS, cosine LR, grad accum)
├── sft/
│   ├── data.py            # ChatML format, loss mask builder
│   ├── trainer.py         # YOUR SFT training loop
│   └── validate.py        # Loss curve comparison vs reference
├── rl/
│   ├── reward.py          # Rule-based reward function
│   ├── trainer.py         # REINFORCE + KL penalty loop
│   └── study_notes.md     # KL instability, reward hacking observations
├── inference/
│   ├── generate.py        # Naive + KV-cache generation
│   └── kvcache.py         # Adapter to existing kvcache/ project
├── ui/                    # nanochat-mlx chat web UI + CLI
├── notebooks/
│   ├── 01_tokenization.ipynb
│   ├── 02_attention.ipynb
│   ├── 03_transformer.ipynb
│   ├── 04_pretraining.ipynb
│   ├── 05_sft.ipynb
│   ├── 06_rl.ipynb
│   └── 07_inference.ipynb
├── benchmarks/
│   ├── loss_curves.py
│   ├── throughput.py      # MPS vs MLX tokens/sec
│   └── eval.py            # Eval scores per checkpoint
├── data/                  # TinyShakespeare, conversation datasets
├── checkpoints/           # gitignored
├── docs/
│   ├── superpowers/specs/ # this file
│   └── study_notes/       # per-stage notes
├── tests/
├── CHALLENGES.md
├── RESUME_ADDITIONS.txt
├── README.md
└── pyproject.toml         # uv-managed
```

---

## 5. Module-level design

### 5.1 `model/config.py`

Two configs - one per track:

```python
@dataclass
class GPTConfig:
    n_layer:    int   = 12
    n_head:     int   = 12
    n_embd:     int   = 768
    vocab_size: int   = 50257
    block_size: int   = 512    # halved from GPT-2's 1024 - fits 16 GB with optimizer states
    dropout:    float = 0.1

# Track 1 (tiny from-scratch on TinyShakespeare)
TINY_CONFIG = GPTConfig(n_layer=6, n_head=6, n_embd=384, block_size=256)  # ~20M params

# Track 2 (GPT-2 checkpoint loading + SFT)
GPT2_CONFIG = GPTConfig(n_layer=12, n_head=12, n_embd=768, block_size=512)  # 124M params
```

### 5.2 `model/attention.py` - MultiHeadAttention

**Data flow:**
```
x: (B, T, C)
→ three linear projections → Q, K, V: each (B, T, C)
→ reshape to (B, n_head, T, head_dim)    where head_dim = C // n_head
→ scores = Q @ K.T / sqrt(head_dim)      shape (B, n_head, T, T)
→ causal mask: set upper triangle to -inf
→ softmax(scores) → attn_weights: (B, n_head, T, T)
→ out = attn_weights @ V                 shape (B, n_head, T, head_dim)
→ reshape to (B, T, C) → output projection
```

**Key decisions:**
- Scale by `√head_dim`: prevents softmax saturation when head_dim is large
- Causal mask registered as buffer (`torch.tril`): moves to device with model, not a learned parameter
- Combined QKV projection (`3*C` out) then split: one matmul instead of three - faster on MPS

**Edge cases:**
- `T > block_size` at inference: slice mask to `[:T, :T]`, assert `T <= block_size`
- `head_dim` must be integer: assert `n_embd % n_head == 0` in `__init__`

### 5.3 `model/blocks.py` - TransformerBlock

```python
def forward(self, x):
    x = x + self.attn(self.ln_1(x))   # Pre-LN attention sub-layer
    x = x + self.mlp(self.ln_2(x))    # Pre-LN feed-forward sub-layer
    return x
```

**Key decisions:**
- Pre-LN (LayerNorm before sub-layer): more stable than Post-LN; no warmup tuning needed
- Residual connections: gradient highway through deep networks
- MLP expansion factor 4× (`n_embd → 4*n_embd → n_embd`): empirically optimal for GPT-class models
- GELU activation (not ReLU): smoother gradient flow; GPT-2 standard

### 5.4 `model/gpt.py` - Full GPT

**Weight tying:** `lm_head.weight = wte.weight`
- Saves ~38M parameters
- Token embedding and output projection share the same vector space

**Forward pass:**
```python
tok_emb = self.wte(idx)                          # (B, T, C)
pos_emb = self.wpe(torch.arange(T, device=dev))  # (T, C)
x = tok_emb + pos_emb
for block in self.blocks:
    x = block(x)
x = self.ln_f(x)
logits = self.lm_head(x)                         # (B, T, vocab_size)
```

**GPT-2 weight loading:** convert HuggingFace checkpoint keys to match this naming scheme. One-time utility script; validate by comparing logits on a sample input.

---

### 5.5 `pretrain/data.py`

**Memory-mapped tokenized file:**
```python
data = np.memmap('train.bin', dtype=np.uint16, mode='r')
ix   = torch.randint(len(data) - block_size, (B,))
x    = torch.stack([torch.from_numpy(data[i  :i+T].astype(np.int64)) for i in ix])
y    = torch.stack([torch.from_numpy(data[i+1:i+T+1].astype(np.int64)) for i in ix])
```

**Why `uint16`:** vocab_size=50257 fits in 2 bytes; halves disk footprint vs int32.

### 5.6 `pretrain/train.py`

**Gradient accumulation (simulates large batches on 16 GB):**
```python
for micro_step in range(grad_accum_steps):
    loss = compute_loss(model, x, y) / grad_accum_steps
    loss.backward()
optimizer.step(); optimizer.zero_grad(set_to_none=True)
```

**Cosine LR with warmup:**
```python
def get_lr(step):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * ratio))
```

**Edge cases:**
- First MPS call is slow (JIT compilation) - warm up before timing benchmarks
- `set_to_none=True` in `zero_grad`: frees memory immediately rather than zeroing

---

### 5.7 `sft/data.py` - ChatML format + loss mask

**Conversation format:**
```
<|im_start|>system\nYou are a helpful assistant.<|im_end|>
<|im_start|>user\nWhat is 2+2?<|im_end|>
<|im_start|>assistant\nIt's 4.<|im_end|>
```

**Loss mask construction:**
```python
loss_mask = np.zeros(seq_len, dtype=np.float32)
for turn in conversation:
    if turn.role == "assistant":
        loss_mask[turn.token_start : turn.token_end] = 1.0
```

**Truncation strategy:** drop from the LEFT (oldest tokens). The most recent assistant turn must always be present in the training window.

**Edge cases:**
1. Multi-turn: mask each assistant turn, not just the last
2. `mask.sum() == 0`: empty assistant turn → skip sample, do not compute loss
3. Sequence longer than `block_size`: truncate left, then re-check mask is nonzero

### 5.8 `sft/trainer.py` - SFT loss (THE DIFFERENTIATOR)

```python
def sft_loss(model, input_ids, loss_mask):
    logits  = model(input_ids)           # (B, T, vocab_size)
    logits  = logits[:, :-1, :]          # (B, T-1, vocab_size)
    targets = input_ids[:, 1:]           # (B, T-1)
    mask    = loss_mask[:, 1:]           # (B, T-1)  ← shift aligns with targets

    per_token = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction='none'
    ).reshape(B, T - 1)

    return (per_token * mask).sum() / mask.sum()   # avg over assistant tokens only
```

**The critical invariant:** divide by `mask.sum()`, not `seq_len`. Long prompts must not dilute the gradient signal.

### 5.9 `sft/validate.py`

1. Load identical checkpoint + data + optimizer state in both your trainer and reference
2. Run 100 steps each, record per-step loss
3. Assert curves within 1e-3 tolerance
4. If diverged: bisect - masking, averaging, or optimizer?

---

### 5.10 `rl/reward.py`

Rule-based reward on TinyShakespeare (deterministic, unit-testable):

```python
def compute_reward(completion: str) -> float:
    reward = 0.0
    if completion.strip().endswith(('.', '!', '?', '"')):
        reward += 0.5
    ngrams = extract_ngrams(completion, n=4)
    repeat_rate = count_repeated(ngrams) / max(len(ngrams), 1)
    reward -= repeat_rate * 0.5
    return float(np.clip(reward, 0.0, 1.0))
```

**Unit tests required before running RL:** perfect completion, empty completion, pure repetition, punctuation-only.

### 5.11 `rl/trainer.py` - REINFORCE + KL penalty

```python
def rl_step(model, ref_model, prompt_tokens, config):
    # 1. sample G completions
    completions = [sample(model, prompt_tokens) for _ in range(config.G)]

    # 2. score + group-relative advantage
    rewards    = torch.tensor([compute_reward(c) for c in completions])
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # 3. policy gradient loss
    pg_loss = sum(-adv * get_log_probs(model, prompt + c).sum()
                  for c, adv in zip(completions, advantages)) / config.G

    # 4. KL penalty to frozen SFT reference
    with torch.no_grad():
        ref_lp = get_log_probs(ref_model, prompt_tokens + completions[0])
    pol_lp = get_log_probs(model, prompt_tokens + completions[0])
    kl = (pol_lp - ref_lp).mean()

    return pg_loss + config.kl_coeff * kl
```

**Starting values:** `G=8`, `kl_coeff=0.1`, `lr=1e-5`.

**Four known failure modes (pre-emptive CHALLENGES.md entries):**
1. All rewards equal → std≈0 → advantage≈0 → no update. Add ε to std.
2. Reward always 0 → check reward function, not the RL loop.
3. KL explodes → increase kl_coeff.
4. NaN loss → model collapsed to empty completions; add length penalty.

---

### 5.12 `inference/generate.py`

**Naive (O(T²)):**
```python
for _ in range(max_new_tokens):
    logits = model(tokens)           # recomputes all prior tokens
    next_token = sample(logits[:, -1, :], temperature, top_k)
    tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
```

**KV-cache (O(T)):**
```python
kv_cache = model.prefill(prompt_tokens)
for _ in range(max_new_tokens):
    logits, kv_cache = model.step(last_token, kv_cache)
    last_token = sample(logits, temperature, top_k)
```

**KV-cache data structure:**
```python
# list of (K, V) per layer
KVCache = list[tuple[torch.Tensor, torch.Tensor]]
# shape per entry: (B, n_head, seq_len_so_far, head_dim)
# grows by 1 in seq_len dimension each step
```

**Edge cases:**
- Cache full at `block_size`: stop generation or implement sliding-window eviction
- `temperature=0`: argmax (greedy, deterministic) - use for benchmarking reproducibility
- MPS JIT warmup: run one dummy generation before timing

### 5.13 `inference/kvcache.py`

Thin adapter that wraps the existing `kvcache/` project's implementation. Validates identical outputs between your existing kvcache and the inline implementation. Documents the integration in `CHALLENGES.md`.

---

## 6. Test strategy

| Test type | What it covers | File |
|---|---|---|
| Round-trip | `decode(encode(s)) == s` for emoji, CJK, code | `tests/test_tokenizer.py` |
| Architecture | `GPT(config)(x).shape == (B, T, vocab_size)` | `tests/test_model.py` |
| Causal mask | position t cannot attend to t+1 | `tests/test_attention.py` |
| Loss mask | masked loss < unmasked loss on prompt-heavy batch | `tests/test_sft.py` |
| SFT sanity | step-0 loss ≈ log(vocab_size) ≈ 10.8 | `tests/test_sft.py` |
| RL reward | unit tests on four known completions | `tests/test_reward.py` |
| KV-cache parity | naive and cached generation produce identical tokens | `tests/test_inference.py` |
| MPS vs MLX | forward pass outputs within 1e-4 | `tests/test_framework_parity.py` |

**TDD order:** write each failing test first, then implement until it passes.

---

## 7. Benchmarks plan

| Benchmark | Metric | Tracks compared |
|---|---|---|
| Pretraining loss curve | train/val loss vs step | Tiny model |
| SFT loss curve | masked loss vs step, vs reference | GPT-2 + your trainer |
| RL reward curve | mean reward vs step | Tiny model |
| KV-cache speedup | tokens/sec naive vs cached | GPT-2 checkpoint |
| MPS vs MLX throughput | tokens/sec forward pass | GPT-2 checkpoint |
| Eval scores | bundled nanochat-mlx evals | base vs SFT vs RL checkpoints |

All benchmark numbers go in the README table. Include machine spec (M2 Pro, 16 GB), batch size, sequence length, and PyTorch/MLX version.

---

## 8. Notebooks plan

| Notebook | Key cell that proves understanding |
|---|---|
| `01_tokenization.ipynb` | Encode/decode round-trip; `" hello" != "hello"` tokenization |
| `02_attention.ipynb` | Implement `scaled_dot_product_attention` from scratch; visualise attention map |
| `03_transformer.ipynb` | Build `TransformerBlock`; verify output shape + residual identity at init |
| `04_pretraining.ipynb` | Loss curve from TinyShakespeare run; gradient norm over steps |
| `05_sft.ipynb` | Show loss mask tensor side-by-side with conversation tokens; before/after masking loss comparison |
| `06_rl.ipynb` | Reward curve; KL divergence plot; one "reward hacking" example |
| `07_inference.ipynb` | Tokens/sec bar chart: naive vs KV-cache; temperature sweep |

---

## 9. Interview answers this project enables

| Question | Component |
|---|---|
| "Implement scaled dot-product attention" | `model/attention.py` |
| "Why scale by √d_k?" | Notebook 02 + attention.py comments |
| "Why residual connections?" | `model/blocks.py` + Notebook 03 |
| "What's wrong with computing loss on prompt tokens?" | `sft/trainer.py` |
| "How does KV-cache reduce generation from O(n²) to O(n)?" | `inference/generate.py` + Notebook 07 |
| "What is group-relative advantage in GRPO?" | `rl/trainer.py` |
| "What does the KL penalty in RL do?" | `rl/trainer.py` + Notebook 06 |
| "Walk me through how ChatGPT is built" | The full project |

---

## 10. Technology choices

| Choice | Reason |
|---|---|
| PyTorch + MPS primary | Frontier labs use PyTorch; MPS runs on Apple Silicon |
| MLX secondary / comparison | Apple-native speed; comparison benchmark is unique in portfolio |
| `uv` package manager | Fast, reproducible; consistent with prior two projects |
| Python 3.11+ | `tomllib` stdlib, match-statement, better error messages |
| TinyShakespeare | Small (~1 MB), fits in RAM, trains in hours, well-understood output |
| GPT-2 small (124M) | Publicly available weights; canonical baseline; fits in 16 GB with optimizer |
| ChatML format | nanochat standard; used by many open models; unambiguous turn boundaries |

---

## 11. Known risks and mitigations

| Risk | Mitigation |
|---|---|
| MPS OOM during SFT | Reduce batch size; enable gradient accumulation; use `block_size=512` |
| SFT loss diverges from reference | Bisect: mask → averaging → optimizer; keep reference in a separate conda env |
| RL NaN loss | Add minimum-length penalty; start with `kl_coeff=0.1`; unit-test reward fn first |
| GPT-2 weight loading mismatch | Write one key-mapping utility; validate logits on a fixed sample input |
| Slow MPS vs MLX | Expected; document honestly in benchmark table; MPS is the learning vehicle |
