import os
import numpy as np
import torch


def get_batch(
    split: str,
    block_size: int,
    batch_size: int,
    device: str,
    data_dir: str = "data",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch from the memory-mapped token file.

    Tokens are stored as uint16 (vocab_size=50257 fits in 2 bytes, halving disk use).
    Cast to int64 at batch time — PyTorch embedding layers require int64.

    Args:
        split: "train" or "val" split name.
        block_size: Context window length (T).
        batch_size: Number of sequences to sample (B).
        device: Target device ("cpu", "cuda", etc.).
        data_dir: Directory containing train.bin and val.bin files.

    Returns:
        (x, y) tensors of shape (B, T), dtype int64, on target device.
        y[b, t] = x[b, t+1] for all t < T-1.
    """
    path = os.path.join(data_dir, f"{split}.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")

    # Sample random start positions, ensuring we don't run past end of file.
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # x: tokens at positions [i, i+1, ..., i+block_size-1]
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )

    # y: tokens at positions [i+1, i+2, ..., i+block_size]
    # This ensures y[b, t] = x[b, t+1] for all t < block_size-1
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + block_size + 1].astype(np.int64)) for i in ix]
    )

    return x.to(device), y.to(device)
