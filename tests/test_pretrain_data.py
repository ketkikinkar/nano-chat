import numpy as np
import torch
import tempfile
import os
import pytest
from pretrain.data import get_batch


def _write_fake_bin(path, n_tokens=10_000):
    arr = np.random.randint(0, 50257, size=n_tokens, dtype=np.uint16)
    arr.tofile(path)


def test_batch_shapes(tmp_path):
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    _write_fake_bin(train_path)
    _write_fake_bin(val_path)
    x, y = get_batch(
        "train",
        block_size=64,
        batch_size=4,
        device="cpu",
        data_dir=str(tmp_path),
    )
    assert x.shape == (4, 64)
    assert y.shape == (4, 64)


def test_y_is_x_shifted_by_one(tmp_path):
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    _write_fake_bin(train_path)
    _write_fake_bin(val_path)
    torch.manual_seed(0)
    x, y = get_batch(
        "train",
        block_size=32,
        batch_size=1,
        device="cpu",
        data_dir=str(tmp_path),
    )
    # y[b, t] must equal x[b, t+1] for all t < T-1
    assert torch.all(x[0, 1:] == y[0, :-1])


def test_dtype_is_int64(tmp_path):
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    _write_fake_bin(train_path)
    _write_fake_bin(val_path)
    x, y = get_batch(
        "train", block_size=16, batch_size=2, device="cpu", data_dir=str(tmp_path)
    )
    assert x.dtype == torch.long
