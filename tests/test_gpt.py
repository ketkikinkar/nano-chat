import torch
import pytest
from model.config import GPTConfig
from model.gpt import GPT

CFG = GPTConfig(n_layer=2, n_head=4, n_embd=64, block_size=16, vocab_size=100)


def test_output_shape():
    model = GPT(CFG)
    idx = torch.randint(0, 100, (2, 8))
    logits = model(idx)
    assert logits.shape == (2, 8, 100)


def test_weight_tying():
    """lm_head and wte must share the exact same tensor object."""
    model = GPT(CFG)
    assert model.lm_head.weight is model.wte.weight


def test_param_count_reasonable():
    """Tiny config should have far fewer than 1M params."""
    model = GPT(CFG)
    n = sum(p.numel() for p in model.parameters())
    assert n < 1_000_000, f"param count {n} unexpectedly large for tiny config"


def test_loss_at_init_near_log_vocab():
    """At random init, cross-entropy loss ≈ log(vocab_size) ≈ 4.6 for vocab=100."""
    torch.manual_seed(0)
    model = GPT(CFG)
    model.eval()
    idx = torch.randint(0, 100, (4, 8))
    logits = model(idx)
    targets = idx[:, 1:].reshape(-1)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 100), targets)
    import math
    expected = math.log(100)  # ~4.60
    assert abs(loss.item() - expected) < 1.5, \
        f"loss at init {loss.item():.2f} too far from log(vocab)={expected:.2f}"
