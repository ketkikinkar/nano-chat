import torch
from model.config import GPTConfig
from model.gpt import GPT
from rl.trainer import get_log_probs, RLConfig

CFG = GPTConfig(n_layer=2, n_head=4, n_embd=64, block_size=16, vocab_size=100)


def test_log_probs_shape():
    model = GPT(CFG)
    ids = torch.randint(0, 100, (8,))
    lp = get_log_probs(model, ids)
    # log prob is a scalar (sum over all tokens)
    assert lp.shape == torch.Size([])


def test_log_probs_negative():
    model = GPT(CFG)
    ids = torch.randint(0, 100, (8,))
    lp = get_log_probs(model, ids)
    assert lp.item() < 0, "log probs must be negative"


def test_rl_config_defaults():
    cfg = RLConfig()
    assert cfg.G == 8
    assert cfg.kl_coeff == 0.1
    assert cfg.lr == 1e-5
