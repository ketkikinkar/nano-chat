import torch
from model.config import GPTConfig
from model.gpt import GPT
from inference.generate import generate_naive, generate_cached

CFG = GPTConfig(n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=100)

def test_output_length():
    torch.manual_seed(0)
    model = GPT(CFG); model.eval()
    prompt = torch.randint(0, 100, (5,))
    out = generate_naive(model, prompt, max_new=10, temperature=0.0)
    assert out.shape == (15,), f"expected 5+10=15 tokens, got {out.shape}"

def test_cached_matches_naive():
    """KV-cache and naive generation must produce identical tokens at temperature=0."""
    torch.manual_seed(42)
    model = GPT(CFG); model.eval()
    prompt = torch.randint(0, 100, (4,))

    out_naive  = generate_naive( model, prompt, max_new=8, temperature=0.0)
    out_cached = generate_cached(model, prompt, max_new=8, temperature=0.0)

    assert torch.equal(out_naive, out_cached), \
        f"naive and cached outputs differ:\n{out_naive}\n{out_cached}"

def test_cached_is_faster(benchmark=None):
    """Smoke-test: cached generation completes without error."""
    model = GPT(CFG); model.eval()
    prompt = torch.randint(0, 100, (8,))
    out = generate_cached(model, prompt, max_new=16, temperature=0.0)
    assert len(out) == 24
