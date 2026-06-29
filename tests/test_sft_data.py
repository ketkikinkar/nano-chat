import torch
import tiktoken
from sft.data import format_conversation, build_batch

ENC = tiktoken.get_encoding("gpt2")

SAMPLE_CONVO = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "What is 2+2?"},
    {"role": "assistant", "content": "It is 4."},
]


def test_format_contains_special_tokens():
    text = format_conversation(SAMPLE_CONVO)
    assert "<|im_start|>" in text
    assert "<|im_end|>" in text
    assert "assistant" in text


def test_loss_mask_only_on_assistant_tokens():
    input_ids, loss_mask = build_batch([SAMPLE_CONVO], ENC, block_size=128)
    # mask must be 1.0 on SOME tokens (the assistant response)
    assert loss_mask.sum() > 0, "loss mask is all zeros — no assistant tokens masked"
    # mask must NOT cover all tokens (prompt tokens should be 0)
    assert loss_mask.sum() < loss_mask.numel(), "loss mask covers everything — prompt not excluded"


def test_loss_mask_dtype():
    input_ids, loss_mask = build_batch([SAMPLE_CONVO], ENC, block_size=128)
    assert loss_mask.dtype == torch.float32


def test_empty_assistant_turn_skipped():
    bad_convo = [
        {"role": "user",      "content": "Hello?"},
        {"role": "assistant", "content": ""},   # empty — should be skipped
    ]
    result = build_batch([bad_convo], ENC, block_size=128)
    assert result is None, "expected None for empty assistant turn"


def test_multi_turn_masks_each_assistant():
    multi = [
        {"role": "user",      "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user",      "content": "Second question"},
        {"role": "assistant", "content": "Second answer"},
    ]
    input_ids, loss_mask = build_batch([multi], ENC, block_size=256)
    # loss sum should be larger than a single-turn conversation
    single = [
        {"role": "user",      "content": "First question"},
        {"role": "assistant", "content": "First answer"},
    ]
    _, mask_single = build_batch([single], ENC, block_size=256)
    assert loss_mask.sum() > mask_single.sum(), "multi-turn should mask more tokens than single-turn"
