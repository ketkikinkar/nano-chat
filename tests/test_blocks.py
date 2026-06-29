import torch
from model.config import GPTConfig
from model.blocks import TransformerBlock

CFG = GPTConfig(n_layer=2, n_head=4, n_embd=64, block_size=16, vocab_size=100)

def test_output_shape():
    block = TransformerBlock(CFG)
    x = torch.randn(2, 8, 64)
    out = block(x)
    assert out.shape == (2, 8, 64)

def test_residual_preserves_input_scale():
    """At init, residual output should be close to input (identity-like)."""
    torch.manual_seed(42)
    block = TransformerBlock(CFG)
    block.eval()
    x = torch.randn(1, 4, 64)
    out = block(x)
    # Not exactly equal, but the scale should be similar
    assert out.std().item() < x.std().item() * 5.0, "output scale exploded"
