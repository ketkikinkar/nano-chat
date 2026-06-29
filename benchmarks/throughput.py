"""Measure tokens/sec: naive vs KV-cache, and MPS vs MLX.
Run: uv run python benchmarks/throughput.py
"""
import time, torch
from model.config import GPT2_CONFIG
from model.gpt import GPT
from inference.generate import generate_naive, generate_cached

PROMPT_LEN  = 32
MAX_NEW     = 64
N_RUNS      = 5
DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu"

def measure(fn, model, prompt, label):
    # Warm up MPS JIT — first call is always slow
    fn(model, prompt, max_new=8, temperature=0.0)
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn(model, prompt, max_new=MAX_NEW, temperature=0.0)
        times.append(time.perf_counter() - t0)
    avg = sum(times) / N_RUNS
    tps = MAX_NEW / avg
    print(f"  {label:<30} {tps:6.1f} tok/s  ({avg*1000:.0f} ms avg over {N_RUNS} runs)")
    return tps

if __name__ == "__main__":
    model = GPT(GPT2_CONFIG).to(DEVICE)
    model.eval()
    prompt = torch.randint(0, 50257, (PROMPT_LEN,)).to(DEVICE)

    print(f"\nThroughput benchmark — {DEVICE} | GPT-2 small (124M) | prompt={PROMPT_LEN} | gen={MAX_NEW}")
    print("-" * 65)
    tps_naive  = measure(generate_naive,  model, prompt, "naive (no cache)")
    tps_cached = measure(generate_cached, model, prompt, "KV-cache")
    print(f"\n  Speedup: {tps_cached/tps_naive:.1f}× (KV-cache vs naive)")
