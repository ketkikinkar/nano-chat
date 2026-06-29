import math
import torch
from model.config import GPTConfig
from model.gpt import GPT
from sft.trainer import sft_loss

CFG = GPTConfig(n_layer=2, n_head=4, n_embd=64, block_size=16, vocab_size=100)

def test_loss_at_init_near_log_vocab():
    """At random init, SFT loss ≈ log(vocab_size) on the masked tokens."""
    torch.manual_seed(0)
    model = GPT(CFG)
    model.eval()
    B, T = 4, 16
    input_ids = torch.randint(0, 100, (B, T))
    # mask the second half of each sequence as "assistant" tokens
    loss_mask = torch.zeros(B, T)
    loss_mask[:, T // 2:] = 1.0
    loss = sft_loss(model, input_ids, loss_mask)
    expected = math.log(100)   # ~4.60
    assert abs(loss.item() - expected) < 1.5, \
        f"SFT loss at init {loss.item():.2f} far from log(vocab)={expected:.2f}"

def test_masked_loss_differs_from_unmasked():
    """Masking only some tokens must change the loss value."""
    torch.manual_seed(1)
    model = GPT(CFG)
    model.eval()
    B, T = 2, 16
    input_ids = torch.randint(0, 100, (B, T))

    full_mask    = torch.ones(B, T)
    partial_mask = torch.zeros(B, T); partial_mask[:, 8:] = 1.0

    loss_full    = sft_loss(model, input_ids, full_mask)
    loss_partial = sft_loss(model, input_ids, partial_mask)
    assert not torch.isclose(loss_full, loss_partial), \
        "masked and unmasked loss must differ"

def test_loss_is_scalar():
    model = GPT(CFG)
    B, T = 2, 16
    ids  = torch.randint(0, 100, (B, T))
    mask = torch.ones(B, T)
    loss = sft_loss(model, ids, mask)
    assert loss.shape == torch.Size([])

def test_all_zero_mask_raises():
    """Zero mask → division by zero → NaN. Implementation must guard this."""
    model = GPT(CFG)
    ids  = torch.randint(0, 100, (2, 16))
    mask = torch.zeros(2, 16)
    loss = sft_loss(model, ids, mask)
    assert not torch.isnan(loss), "all-zero mask produced NaN — add guard"
