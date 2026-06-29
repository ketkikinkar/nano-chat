import torch
import pytest
from model.config import GPTConfig
from model.attention import MultiHeadAttention

# Small config for speed in all tests
CFG = GPTConfig(n_layer=2, n_head=4, n_embd=64, block_size=16, vocab_size=100)


def test_output_shape():
    attn = MultiHeadAttention(CFG)
    x = torch.randn(2, 8, 64)
    out = attn(x)
    assert out.shape == (2, 8, 64)


def test_causal_mask_blocks_future():
    """Changing token at position 3 must not affect outputs at positions 0-2."""
    torch.manual_seed(0)
    attn = MultiHeadAttention(CFG)
    attn.eval()
    x = torch.randn(1, 6, 64)
    x_mod = x.clone()
    x_mod[0, 5, :] += 999.0          # large change at last position
    out1 = attn(x)
    out2 = attn(x_mod)
    # positions 0..4 must be unaffected
    assert torch.allclose(out1[0, :5, :], out2[0, :5, :], atol=1e-5), \
        "causal mask broken: future token leaked into past positions"


def test_head_dim_assertion():
    with pytest.raises(AssertionError):
        bad_cfg = GPTConfig(n_embd=65, n_head=4)  # 65 not divisible by 4
