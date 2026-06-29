from dataclasses import dataclass


@dataclass
class GPTConfig:
    n_layer:    int   = 12
    n_head:     int   = 12
    n_embd:     int   = 768
    vocab_size: int   = 50257
    block_size: int   = 512
    dropout:    float = 0.1

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, \
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"


# ~20M params: fast to train on TinyShakespeare in a few hours on M2 Pro
TINY_CONFIG = GPTConfig(n_layer=6, n_head=6, n_embd=384, block_size=256, vocab_size=50257)

# GPT-2 small adapted: halved block_size to fit 16 GB with optimizer states
GPT2_CONFIG = GPTConfig(n_layer=12, n_head=12, n_embd=768, block_size=512, vocab_size=50257)
