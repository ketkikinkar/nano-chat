from model.config import GPTConfig, TINY_CONFIG, GPT2_CONFIG


def test_default_config_fields():
    cfg = GPTConfig()
    assert cfg.n_layer == 12
    assert cfg.n_head == 12
    assert cfg.n_embd == 768
    assert cfg.vocab_size == 50257
    assert cfg.block_size == 512
    assert cfg.dropout == 0.1


def test_head_dim_divides_evenly():
    cfg = GPTConfig(n_embd=64, n_head=4)
    assert cfg.n_embd % cfg.n_head == 0


def test_tiny_config():
    assert TINY_CONFIG.n_layer == 6
    assert TINY_CONFIG.n_embd == 384
    assert TINY_CONFIG.block_size == 256


def test_gpt2_config():
    assert GPT2_CONFIG.n_layer == 12
    assert GPT2_CONFIG.n_embd == 768
    assert GPT2_CONFIG.block_size == 512
