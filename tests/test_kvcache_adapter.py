"""Tests for the kvcache adapter that connects GPT to the external kvcache project."""
import sys
import os

# Make the kvcache project importable — it lives as a sibling to nanochat
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../kvcache"))

import torch
from model.config import GPTConfig
from model.gpt import GPT
from inference.kvcache import generate_with_eviction

CFG = GPTConfig(n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=100)


def test_eviction_output_length():
    """generate_with_eviction must return exactly prompt_len + max_new tokens."""
    model = GPT(CFG)
    model.eval()
    prompt = torch.randint(0, 100, (4,))
    out = generate_with_eviction(model, prompt, max_new=8, budget_fraction=0.5)
    assert len(out) == 12
