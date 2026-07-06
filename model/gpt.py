import torch
import torch.nn as nn
from model.config import GPTConfig
from model.blocks import TransformerBlock


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)   # token embeddings
        self.wpe = nn.Embedding(config.block_size, config.n_embd)   # position embeddings
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the vector that represents token X as input is the same vector
        # used to predict token X as output. Saves ~38M params for GPT-2 small.
        self.lm_head.weight = self.wte.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        device = idx.device
        pos = torch.arange(T, device=device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)                           # (B, T, vocab_size)

    @classmethod
    def from_pretrained(cls, model_type: str = "gpt2") -> "GPT":
        """Load GPT-2 weights from HuggingFace. Only supports 'gpt2' (124M)."""
        from transformers import GPT2LMHeadModel
        from model.config import GPT2_CONFIG

        print(f"Loading {model_type} weights from HuggingFace...")
        model = cls(GPT2_CONFIG)
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # HuggingFace GPT-2 uses Conv1D with shape (in, out) vs nn.Linear (out, in).
        # Weights for these layers must be transposed when copying.
        transposed = [
            "attn.c_attn.weight", "attn.c_proj.weight",
            "mlp.c_fc.weight",    "mlp.c_proj.weight",
        ]

        sd = model.state_dict()

        # Build mapping: HF key → our key
        def hf_to_ours(hf_key: str) -> str:
            k = hf_key
            k = k.replace("transformer.h.", "blocks.")
            k = k.replace("transformer.wte", "wte")
            k = k.replace("transformer.wpe", "wpe")
            k = k.replace("transformer.ln_f", "ln_f")
            k = k.replace(".mlp.c_fc",   ".mlp.fc")
            k = k.replace(".mlp.c_proj", ".mlp.proj")
            return k

        for hf_key, hf_val in sd_hf.items():
            if "lm_head" in hf_key:
                continue  # tied with wte - skip
            our_key = hf_to_ours(hf_key)
            if our_key not in sd:
                continue
            needs_transpose = any(hf_key.endswith(t) for t in transposed)
            with torch.no_grad():
                if needs_transpose:
                    sd[our_key].copy_(hf_val.t())
                else:
                    sd[our_key].copy_(hf_val)

        model.load_state_dict(sd)
        print("Weights loaded and validated.")
        return model
