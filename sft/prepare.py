"""Run once: uv run python sft/prepare.py

Downloads the Stanford Alpaca dataset — the original 52K instruction-following
examples used to fine-tune LLaMA into Alpaca. Stored locally so training can
run offline and deterministically after the one-time download.
"""
import os, requests, json

URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"

def main():
    os.makedirs("data", exist_ok=True)
    path = "data/alpaca.json"
    if os.path.exists(path):
        print(f"Already exists: {path}")
        return
    print("Downloading Alpaca dataset...")
    r = requests.get(URL)
    with open(path, "w") as f:
        f.write(r.text)
    data = json.loads(r.text)
    print(f"Downloaded {len(data)} entries → {path}")

if __name__ == "__main__":
    main()
