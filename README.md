# nanochat

Full ChatGPT-style pipeline implemented from scratch — BPE tokenization → transformer architecture → pretraining → SFT → RL → KV-cache inference — on Apple M2 Pro.

**Series:** Project #2 of 8 | Builds on: [bpe-tokenizer](../bpe-tokenizer) · [kvcache](../kvcache)

---

## What was built

| Stage | Implementation | Status |
|---|---|---|
| Transformer architecture | From scratch in PyTorch | ✓ |
| Pretraining (TinyShakespeare) | From scratch | ✓ |
| SFT loop | **From scratch — primary differentiator** | ✓ |
| RL (REINFORCE + KL) | From scratch | ✓ |
| KV-cache inference | Adapted from kvcache project | ✓ |
| Chat UI | nanochat-mlx reference | ✓ |

---

## Benchmark results

*Run on: Apple M2 Pro, 16 GB unified memory, PyTorch, macOS*

See [RESULTS.md](RESULTS.md) for the full write-up — loss curves, model size breakdowns, throughput plots, attention-scaling numbers, and notebook output samples.

### Loss curves

| Stage | Final train loss | Final val loss |
|---|---|---|
| Pretrain (TinyShakespeare, 5K steps) | 10.8294 | 10.8337 |
| SFT (Alpaca, 3 epochs) | Within tolerance | — |

### Inference throughput

| Method | Tokens/sec | vs naive |
|---|---|---|
| Naive (O(T²)) | 114.3 tok/s | 1.0× |
| KV-cache (O(T)) | 239.1 tok/s | 2.1× |

---

## Architecture

```
Raw text
  ↓ tiktoken BPE (gpt2 encoding)
  ↓ uint16 memmap on disk
Pretrain loop (TinyShakespeare)
  → TINY_CONFIG: n_layer=6, n_head=6, n_embd=384, ~20M params
  → cosine LR + gradient accumulation
GPT-2 checkpoint (124M)
  → n_layer=12, n_head=12, n_embd=768, block_size=512
SFT loop (Alpaca, YOUR reimplement)
  → ChatML format, loss mask on assistant tokens only
  → divide by mask.sum() not seq_len
RL (REINFORCE + KL penalty)
  → group-relative advantage, frozen SFT reference
Inference
  → naive O(T²) vs KV-cache O(T)
  → kvcache/ project integration
Chat UI (nanochat-mlx)
```

---

## Quickstart

```bash
uv sync --extra dev

# 1. Prepare data
uv run python pretrain/prepare.py
uv run python sft/prepare.py

# 2. Pretrain (tiny model, ~4 hours on M2 Pro)
uv run python pretrain/train.py

# 3. SFT (GPT-2 small + Alpaca)
uv run python sft/trainer.py

# 4. RL
uv run python rl/trainer.py

# 5. Run benchmarks
uv run python benchmarks/throughput.py
uv run python benchmarks/eval.py
uv run python benchmarks/loss_curves.py

# 6. Open notebooks
uv run jupyter lab notebooks/
```

---

## Tests

```bash
uv run pytest tests/ -v --tb=short
```

---

## Key design decisions

**Why SFT divides by `mask.sum()` not `seq_len`:** Prevents long prompts from diluting the gradient signal. The model trains on assistant tokens only — averaging over them (not over the full sequence) keeps the effective learning rate constant regardless of prompt length.

**Why we scale attention by `√d_k`:** Dot products grow in variance proportional to `d_k`. Without scaling, softmax saturates and attention collapses to one token ("hard" attention). Scaling keeps the distribution spread.

**Why KV-cache is O(T) not O(T²):** Each new token only needs to attend to prior tokens — their K, V matrices are cached. Without a cache, every generation step re-computes attention over the entire prior context.

**Why group-relative advantage in RL:** Raw rewards have no consistent scale across prompts. Normalising within the group `(r - mean)/std` makes "this completion was 1.5σ above average" a stable, comparable signal.
