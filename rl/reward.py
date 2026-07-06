import numpy as np


def extract_ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    """Extract n-grams from text.

    Splits text into words and returns tuples of n consecutive words.
    Used to detect repetitive patterns in model outputs.
    """
    words = text.lower().split()
    return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]


def compute_reward(completion: str) -> float:
    """Rule-based reward for TinyShakespeare completions.

    Deterministic and unit-testable - avoids needing a trained reward model.
    Returns a float in [0, 1].

    Rewards proper sentence endings (+0.5) and reasonable length (+0.2).
    Penalizes 4-gram repetition (up to -0.5 for full degeneration).
    """
    if not completion.strip():
        return 0.0

    reward = 0.0

    # Reward: proper sentence endings signal coherent text
    if completion.strip().endswith(('.', '!', '?', '"', "'")):
        reward += 0.5

    # Reward: reasonable length (not too short)
    words = completion.split()
    if len(words) >= 5:
        reward += 0.2

    # Penalty: 4-gram repetition is a known failure mode of small LMs (reward hacking)
    ngrams = extract_ngrams(completion, n=4)
    if ngrams:
        unique_rate = len(set(ngrams)) / len(ngrams)
        reward -= (1.0 - unique_rate) * 0.5   # penalise up to 0.5 for full repetition

    return float(np.clip(reward, 0.0, 1.0))
