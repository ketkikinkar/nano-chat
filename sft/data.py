from __future__ import annotations

import numpy as np
import torch
from tiktoken import Encoding

# ChatML special tokens — used by nanochat and many open models (e.g. ChatGPT, Mistral)
IM_START = "<|im_start|>"
IM_END   = "<|im_end|>"


def format_conversation(turns: list[dict]) -> str:
    """Format a list of {role, content} dicts into a ChatML string.

    Each turn becomes: <|im_start|>role\ncontent<|im_end|>\n
    This is the de-facto standard for instruction-tuned models — structuring
    the dialogue so the model learns role boundaries from the tokens themselves.
    """
    parts = []
    for turn in turns:
        parts.append(f"{IM_START}{turn['role']}\n{turn['content']}{IM_END}\n")
    return "".join(parts)


def build_batch(
    conversations: list[list[dict]],
    tokenizer: Encoding,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Tokenize conversations and build the loss mask.

    Returns (input_ids, loss_mask) of shape (B, block_size), or None if any
    conversation has no assistant tokens (all-zero mask would produce NaN loss
    because the cross-entropy denominator would be zero).
    """
    all_ids, all_masks = [], []

    for turns in conversations:
        ids, mask = _encode_with_mask(turns, tokenizer, block_size)
        if ids is None:
            # Bail out early — a NaN-producing sample contaminates the whole batch
            return None
        all_ids.append(ids)
        all_masks.append(mask)

    return (
        torch.tensor(np.stack(all_ids),   dtype=torch.long),
        torch.tensor(np.stack(all_masks), dtype=torch.float32),
    )


def _encode_with_mask(
    turns: list[dict],
    tokenizer: Encoding,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Encode one conversation and return (token_ids, loss_mask).

    The loss mask is 1.0 only on assistant *content* tokens — the model should
    only be penalised for what it generates, not for the user prompt or the
    ChatML scaffolding tokens (headers / footers).
    """
    ids:  list[int]   = []
    mask: list[float] = []

    for turn in turns:
        header  = f"{IM_START}{turn['role']}\n"
        content = turn["content"]
        footer  = IM_END + "\n"

        h_ids = tokenizer.encode_ordinary(header)
        c_ids = tokenizer.encode_ordinary(content)
        f_ids = tokenizer.encode_ordinary(footer)

        is_assistant = (turn["role"] == "assistant")

        # Headers and footers are structural scaffolding — never compute loss on them
        h_mask = [0.0] * len(h_ids)
        # Only grade the model on its own outputs (assistant turns)
        c_mask = [1.0 if is_assistant else 0.0] * len(c_ids)
        f_mask = [0.0] * len(f_ids)

        ids  += h_ids + c_ids + f_ids
        mask += h_mask + c_mask + f_mask

    # Guard: if no assistant content exists, loss = 0/0 = NaN — skip this sample
    if sum(mask) == 0:
        return None, None

    # Truncate from the LEFT so the most recent assistant turn is always present.
    # Left-truncation preserves the end of the conversation where the last answer lives.
    if len(ids) > block_size:
        ids  = ids[-block_size:]
        mask = mask[-block_size:]

    # Pad to block_size with token id 0 and mask 0.0 so batches are uniform shape
    pad_len = block_size - len(ids)
    ids  = ids  + [0] * pad_len
    mask = mask + [0.0] * pad_len

    return np.array(ids, dtype=np.int64), np.array(mask, dtype=np.float32)
