"""Run once: python pretrain/prepare.py — downloads TinyShakespeare, tokenizes to data/."""
import os
import requests
import tiktoken
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = "data"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_path = os.path.join(DATA_DIR, "shakespeare.txt")

    # Download TinyShakespeare if not already present.
    if not os.path.exists(raw_path):
        print("Downloading TinyShakespeare...")
        r = requests.get(DATA_URL)
        with open(raw_path, "w") as f:
            f.write(r.text)
        print(f"  {len(r.text):,} chars")

    # Load raw text and tokenize with GPT-2 BPE encoder.
    text = open(raw_path).read()
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(text)
    print(f"Tokenized: {len(ids):,} tokens")

    # Split 90/10 train/val and save as uint16 binary files.
    split = int(0.9 * len(ids))
    train_ids = np.array(ids[:split], dtype=np.uint16)
    val_ids = np.array(ids[split:], dtype=np.uint16)
    train_ids.tofile(os.path.join(DATA_DIR, "train.bin"))
    val_ids.tofile(os.path.join(DATA_DIR, "val.bin"))
    print(f"train: {len(train_ids):,} tokens → data/train.bin")
    print(f"val:   {len(val_ids):,} tokens  → data/val.bin")


if __name__ == "__main__":
    main()
